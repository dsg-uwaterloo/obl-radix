#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "external/radix_partition/data-types.h"
#include "carry_forward.h"
#include "oblivious_ops.h"
#include "prefix_sum_expand.h"
#include "slice_utils.h"

extern "C" {
#include "bitonic.h"
#include "threading.h"
}

namespace pad_table_sort_detail {

constexpr double kSecParam = 40.0;
constexpr double kInvE = 0.36787944117144233; // exp(-1)
constexpr double kE = 2.71828182845904509;    // exp(1)

inline std::uint32_t bool_to_u32(bool x) {
  return static_cast<std::uint32_t>(x);
}

inline std::uint32_t ct_mask_u32(std::uint32_t sel01) { return 0u - sel01; }

inline std::uint32_t ct_select_u32(std::uint32_t a, std::uint32_t b,
                                   std::uint32_t sel01) {
  const std::uint32_t mask = ct_mask_u32(sel01); // 0x00000000 or 0xffffffff
  return (a & ~mask) | (b & mask);
}

inline bool is_power_of_two(std::uint32_t x) {
  return x && ((x & (x - 1u)) == 0);
}

inline std::uint32_t log2_pow2(std::uint32_t x) {
  return static_cast<std::uint32_t>(__builtin_ctz(x));
}

inline std::uint32_t mask_lowbits(std::uint32_t bits) {
  if (bits >= 32u)
    return 0xffffffffu;
  if (bits == 0u)
    return 0u;
  return (1u << bits) - 1u;
}

inline double lambert_w0(double x) {
  const double lower = -kInvE;
  if (x < lower)
    x = lower + 1e-12;
  double w = (x < 1.0) ? x : std::log(x);
  if (!std::isfinite(w))
    w = 0.0;
  for (int i = 0; i < 32; ++i) {
    double e = std::exp(w);
    double f = w * e - x;
    double denom = e * (w + 1.0) - (w + 2.0) * f / (2.0 * w + 2.0);
    if (std::fabs(denom) < 1e-18)
      denom = (denom >= 0 ? 1e-18 : -1e-18);
    double delta = f / denom;
    w -= delta;
    if (std::fabs(delta) <= 1e-12)
      break;
  }
  return w;
}

// Used to compute a public fixed per-bucket capacity.
inline std::uint32_t elemPerBucket(std::uint64_t totalSize,
                                   std::uint64_t bucketCount) {
  if (bucketCount == 0 || totalSize == 0)
    return 0;
  const double mu =
      static_cast<double>(totalSize) / static_cast<double>(bucketCount);
  if (mu <= 0.0)
    return 0;
  double alpha =
      std::log(static_cast<double>(bucketCount) * std::pow(2.0, kSecParam));
  double rhs = alpha / (kE * mu) - kInvE;
  rhs = std::max(rhs, -kInvE + 1e-12);
  const double epsilon = std::pow(kE, lambert_w0(rhs) + 1.0) - 1.0;
  double b = mu * (1.0 + epsilon);
  if (!std::isfinite(b) || b < 1.0)
    b = 1.0;
  return static_cast<std::uint32_t>(std::ceil(b));
}

inline void run_bitonic_hashkey_sort(row_t *rows, std::uint32_t n,
                                     std::uint32_t numThreads,
                                     std::uint32_t totalBits) {
  numThreads = std::max<std::uint32_t>(1u, numThreads);
  if (n <= 1u)
    return;

  thread_system_init();
  std::vector<std::thread> pool;
  pool.reserve(numThreads > 0 ? numThreads - 1u : 0u);
  for (std::uint32_t i = 1; i < numThreads; ++i)
    pool.emplace_back(thread_start_work);

  bitonic_sort_hashkey_lowbits_(reinterpret_cast<elem_t *>(rows), true, 0,
                                static_cast<int>(n),
                                static_cast<int>(numThreads), totalBits);

  thread_release_all();
  for (auto &t : pool)
    t.join();
  thread_system_cleanup();
}

class BitonicWorkerPool {
public:
  explicit BitonicWorkerPool(std::uint32_t numThreads)
      : numThreads_(std::max<std::uint32_t>(1u, numThreads)) {}

  BitonicWorkerPool(const BitonicWorkerPool &) = delete;
  BitonicWorkerPool &operator=(const BitonicWorkerPool &) = delete;

  ~BitonicWorkerPool() { stop(); }

  void start() {
    if (started_)
      return;
    thread_system_init();
    workers_.reserve(numThreads_ > 0 ? numThreads_ - 1u : 0u);
    for (std::uint32_t i = 1; i < numThreads_; ++i)
      workers_.emplace_back(thread_start_work);
    started_ = true;
  }

  void stop() {
    if (!started_)
      return;
    thread_release_all();
    for (auto &t : workers_)
      t.join();
    workers_.clear();
    thread_system_cleanup();
    started_ = false;
  }

  void sort_hashkey_lowbits(row_t *rows, std::uint32_t n,
                            std::uint32_t totalBits) {
    if (n <= 1u)
      return;
    start();
    bitonic_sort_hashkey_lowbits_(reinterpret_cast<elem_t *>(rows), true, 0,
                                  static_cast<int>(n),
                                  static_cast<int>(numThreads_), totalBits);
  }

  std::uint32_t numThreads() const { return numThreads_; }

private:
  std::uint32_t numThreads_;
  bool started_ = false;
  std::vector<std::thread> workers_;
};

template <class Func>
inline void parallel_slices(std::uint32_t n, std::uint32_t numThreads,
                            const Func &fn) {
  numThreads = std::max<std::uint32_t>(1u, numThreads);
  const std::uint32_t threads =
      std::max<std::uint32_t>(1u, std::min<std::uint32_t>(numThreads, n));
  const auto slices = buildSlices(n, threads);

  if (threads == 1u) {
    fn(slices[0]);
    for (std::size_t i = 1; i < slices.size(); ++i)
      fn(slices[i]);
    return;
  }

  std::vector<std::thread> pool;
  pool.reserve(threads - 1u);
  for (std::uint32_t t = 0; t + 1u < threads; ++t) {
    const Slice sl = slices[t];
    pool.emplace_back([&, sl] { fn(sl); });
  }
  fn(slices[threads - 1u]);
  for (std::size_t t = threads; t < slices.size(); ++t)
    fn(slices[t]);
  for (auto &th : pool)
    th.join();
}

// Parallel run-length encoding over bucketId = (hashKey & lowMask), writing the
// run length (including the per-bucket header row) to cntExpand at the end of
// each run. Other positions have cntExpand==0. This mirrors the structure of
// parallelCounts() but uses bucketId instead of key and uses cntExpand as a
// scratch field on `rows`.
inline void bucketRunCountsParallel(row_t *rows, std::uint32_t n,
                                    std::uint32_t lowMask,
                                    std::uint32_t numThreads) {
  if (n == 0)
    return;
  numThreads = std::max<std::uint32_t>(1u, numThreads);
  const auto slices = buildSlices(n, numThreads);
  const std::size_t P = slices.size();
  if (P == 0)
    return;

  std::vector<std::uint32_t> lastLen(P, 0);
  std::vector<std::uint32_t> mergeVal(P > 0 ? P - 1 : 0, 0);

  // Pass 1: local run-lengths per slice.
  {
    std::vector<std::thread> pool;
    pool.reserve(P);
    for (std::size_t t = 0; t < P; ++t) {
      pool.emplace_back([&, t] {
        const Slice sl = slices[t];
        if (sl.begin >= sl.end) {
          lastLen[t] = 0;
          return;
        }

        std::uint32_t prev = rows[sl.begin].hashKey & lowMask;
        std::uint32_t r = 1u;
        std::uint32_t i = sl.begin + 1u;
        const std::uint32_t end = sl.end;

        // Initialize all cntExpand within the slice to 0; then only run-ends
        // will receive a non-zero count.
        rows[sl.begin].cntExpand = 0u;

        for (; i < end; ++i) {
          const std::uint32_t curr = rows[i].hashKey & lowMask;
          const std::uint32_t match = 0u - bool_to_u32(curr == prev);
          // If run ends at i-1, store its length there.
          rows[i - 1u].cntExpand = (~match) & r;
          r = (match & (r + 1u)) | (~match & 1u);
          prev = (match & prev) | (~match & curr);
          rows[i].cntExpand = 0u;
        }
        rows[end - 1u].cntExpand = r;
        lastLen[t] = r;
      });
    }
    for (auto &th : pool)
      th.join();
  }

  if (P < 2)
    return;

  // Pass 2: merge values across slice boundaries.
  {
    std::vector<std::thread> pool;
    pool.reserve(P - 1);
    for (std::size_t t = 1; t < P; ++t) {
      pool.emplace_back([&, t] {
        const std::uint32_t first = rows[slices[t].begin].hashKey & lowMask;
        const std::uint32_t last =
            rows[slices[t - 1].end - 1u].hashKey & lowMask;
        const std::uint32_t match = 0u - bool_to_u32(first == last);
        mergeVal[t - 1u] = match & lastLen[t - 1u];
        const std::size_t idx = static_cast<std::size_t>(slices[t - 1].end) - 1u;
        rows[idx].cntExpand = (~match) & rows[idx].cntExpand;
      });
    }
    for (auto &th : pool)
      th.join();
  }

  if (P < 3)
    goto finalize;

  // Pass 3: accumulate merge values for runs spanning more than 2 slices.
  {
    std::vector<std::thread> pool;
    pool.reserve(P - 2);
    for (std::size_t i = 2; i < P; ++i) {
      pool.emplace_back([&, i] {
        const std::uint32_t first = rows[slices[i].begin].hashKey & lowMask;
        std::uint32_t acc = 0;
        for (std::size_t j = 0; j < i - 1; ++j) {
          const std::uint32_t last =
              rows[slices[j].end - 1u].hashKey & lowMask;
          const std::uint32_t match = 0u - bool_to_u32(first == last);
          acc += match & lastLen[j];
        }
        mergeVal[i - 1u] += acc;
      });
    }
    for (auto &th : pool)
      th.join();
  }

finalize:
  // Pass 4: add mergeVal[t-1] to the first run-end within slice t.
  {
    std::vector<std::thread> pool;
    pool.reserve(P - 1);
    for (std::size_t t = 1; t < P; ++t) {
      pool.emplace_back([&, t] {
        const Slice sl = slices[t];
        std::uint32_t done = 0u;
        const std::uint32_t mv = mergeVal[t - 1u];
        for (std::uint32_t j = sl.begin; j < sl.end; ++j) {
          const std::uint32_t isNZ = 0u - bool_to_u32(rows[j].cntExpand != 0u);
          const std::uint32_t doAdd = (~done) & isNZ;
          rows[j].cntExpand += doAdd & mv;
          done |= doAdd;
        }
      });
    }
    for (auto &th : pool)
      th.join();
  }
}

} // namespace pad_table_sort_detail

// Shared context that caches all padding setup that depends only on:
// - p = 2^(NUM_RADIX_BITS + log2(bins))
// - numThreads
//
// U/need/L depend on the specific input size N and are computed per call.
class PadTableSortContext {
public:
  PadTableSortContext(std::uint32_t bins, std::uint32_t numThreads)
      : bins_(bins), numThreads_(std::max<std::uint32_t>(1u, numThreads)),
        pool_(numThreads_) {
    using namespace pad_table_sort_detail;
    if (!is_power_of_two(bins_))
      throw std::runtime_error("PadTableSortContext: bins must be power-of-two");

    constexpr std::uint32_t rBits = NUM_RADIX_BITS;
    const std::uint32_t bBits = log2_pow2(bins_);
    totalBits_ = rBits + bBits;
    if (totalBits_ >= 31u)
      throw std::runtime_error("PadTableSortContext: totalBits too large");

    lowMask_ = mask_lowbits(totalBits_);
    p_ = 1u << totalBits_;

    // Pre-build one header row per bucket. This is reused for R and S.
    bucketHeaders_.resize(p_);
    std::memset(bucketHeaders_.data(), 0,
                static_cast<std::size_t>(p_) * sizeof(row_t));
    pad_table_sort_detail::parallel_slices(p_, numThreads_, [&](const Slice &sl) {
      const std::uint32_t maxIdx = std::numeric_limits<std::uint32_t>::max();
      for (std::uint32_t i = sl.begin; i < sl.end; ++i) {
        bucketHeaders_[i].idx = maxIdx;
        bucketHeaders_[i].hashKey = i & lowMask_;
      }
    });

    // Scratch that depends only on p.
    headers_.resize(p_);
    selected_.reset(new bool[p_]);
  }

  PadTableSortContext(const PadTableSortContext &) = delete;
  PadTableSortContext &operator=(const PadTableSortContext &) = delete;

  void pad(table_t &tbl) {
    using namespace pad_table_sort_detail;

    if (tbl.num_tuples == 0)
      return;

    const std::uint64_t N64 = tbl.num_tuples;
    if (N64 >
        static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()))
      throw std::runtime_error("padTableSort: input too large for current impl");
    const std::uint32_t N = static_cast<std::uint32_t>(N64);

    const std::uint32_t U =
        std::max<std::uint32_t>(1u, elemPerBucket(N64, p_));
    const std::uint64_t finalN64 = static_cast<std::uint64_t>(p_) * U;
    if (finalN64 < N64)
      throw std::runtime_error("padTableSort: overflow in finalN");
    const std::uint64_t need64 = finalN64 - N64;
    if (need64 == 0)
      return;
    if (need64 > static_cast<std::uint64_t>(
                     std::numeric_limits<std::uint32_t>::max()))
      throw std::runtime_error("padTableSort: padding too large");
    const std::uint32_t need = static_cast<std::uint32_t>(need64);

    const std::uint32_t M = N + p_;
    if (work_.size() < static_cast<std::size_t>(M))
      work_.resize(M);
    if (!isHeader_ || isHeaderCap_ < M) {
      isHeader_.reset(new bool[M]);
      isHeaderCap_ = M;
    }

    // work = original rows + per-bucket headers.
    std::memcpy(work_.data(), tbl.tuples,
                static_cast<std::size_t>(N) * sizeof(row_t));
    std::memcpy(work_.data() + static_cast<std::size_t>(N),
                bucketHeaders_.data(),
                static_cast<std::size_t>(p_) * sizeof(row_t));

    pool_.sort_hashkey_lowbits(work_.data(), M, totalBits_);

    bucketRunCountsParallel(work_.data(), M, lowMask_, numThreads_);
    parallel_slices(M, numThreads_, [&](const Slice &sl) {
      const std::uint32_t maxIdx = std::numeric_limits<std::uint32_t>::max();
      for (std::uint32_t i = sl.begin; i < sl.end; ++i) {
        const std::uint32_t isHeader = bool_to_u32(work_[i].idx == maxIdx);
        const std::uint32_t runLen = work_[i].cntExpand;
        const std::uint32_t realCount = runLen - 1u;
        work_[i].cntExpand =
            ct_select_u32(work_[i].cntExpand, realCount, isHeader);
      }
    });

    parallel_slices(M, numThreads_, [&](const Slice &sl) {
      const std::uint32_t maxIdx = std::numeric_limits<std::uint32_t>::max();
      for (std::uint32_t i = sl.begin; i < sl.end; ++i)
        isHeader_[i] = (work_[i].idx == maxIdx);
    });
    obli_compact_rows(work_.data(), isHeader_.get(), M, numThreads_);

    std::memcpy(headers_.data(), work_.data(),
                static_cast<std::size_t>(p_) * sizeof(row_t));

    parallel_slices(p_, numThreads_, [&](const Slice &sl) {
      for (std::uint32_t i = sl.begin; i < sl.end; ++i) {
        const std::uint32_t real = headers_[i].cntExpand;
        headers_[i].cntExpand = U - real;
      }
    });

    table_t hdrTbl;
    hdrTbl.tuples = headers_.data();
    hdrTbl.num_tuples = p_;
    const auto slicesHdr = buildSlices(p_, numThreads_);
    const std::uint64_t computedNeed64 =
        prefixSumExpandParallel(hdrTbl, slicesHdr, selected_.get());
    if (computedNeed64 != need64)
      throw std::runtime_error("padTableSort: prefix sum mismatch");

    const std::uint32_t L = std::max<std::uint32_t>(p_, need);
    if (dummyWork_.size() < static_cast<std::size_t>(L))
      dummyWork_.resize(L);

    // Bulk zero-fill then set the one non-zero dummy tag field (idx).
    std::memset(dummyWork_.data(), 0, static_cast<std::size_t>(L) * sizeof(row_t));
    parallel_slices(L, numThreads_, [&](const Slice &sl) {
      const std::uint32_t maxIdx = std::numeric_limits<std::uint32_t>::max();
      for (std::uint32_t i = sl.begin; i < sl.end; ++i)
        dummyWork_[i].idx = maxIdx;
    });

    parallel_slices(p_, numThreads_, [&](const Slice &sl) {
      for (std::uint32_t i = sl.begin; i < sl.end; ++i) {
        row_t m = headers_[i];
        const std::uint32_t needHere = headers_[i].cntExpand;
        const std::uint32_t keep = bool_to_u32(needHere != 0u);
        m.cntExpand = ct_select_u32(0u, 1u, keep);
        m.hashKey &= lowMask_;
        m.idx = ct_select_u32(std::numeric_limits<std::uint32_t>::max(), m.idx,
                              keep);
        m.cntSelf = 0u;
        m.key = 0u;
        m.shuffledIdx = 0u;
        std::memset(m.pay, 0, sizeof(m.pay));
        dummyWork_[i] = m;
      }
    });

    obli_distribute_rows(dummyWork_.data(), L, numThreads_);

    table_t dummyTbl;
    dummyTbl.tuples = dummyWork_.data();
    dummyTbl.num_tuples = need;
    carryForwardParallel(dummyTbl, buildSlices(need, numThreads_));

    parallel_slices(need, numThreads_, [&](const Slice &sl) {
      for (std::uint32_t i = sl.begin; i < sl.end; ++i) {
        dummyWork_[i].idx = std::numeric_limits<std::uint32_t>::max();
        dummyWork_[i].cntSelf = 0u;
        dummyWork_[i].cntExpand = 0u;
        dummyWork_[i].hashKey &= lowMask_;
        dummyWork_[i].key = 0u;
        dummyWork_[i].shuffledIdx = 0u;
        std::memset(dummyWork_[i].pay, 0, sizeof(dummyWork_[i].pay));
      }
    });

    row_t *expanded = new row_t[N64 + need64];
    std::memcpy(expanded, tbl.tuples,
                static_cast<std::size_t>(N64) * sizeof(row_t));
    std::memcpy(expanded + static_cast<std::size_t>(N64), dummyWork_.data(),
                static_cast<std::size_t>(need) * sizeof(row_t));
    delete[] tbl.tuples;
    tbl.tuples = expanded;
    tbl.num_tuples = N64 + need64;
  }

private:
  std::uint32_t bins_;
  std::uint32_t numThreads_;
  std::uint32_t totalBits_ = 0;
  std::uint32_t lowMask_ = 0;
  std::uint32_t p_ = 0;

  pad_table_sort_detail::BitonicWorkerPool pool_;

  std::vector<row_t> bucketHeaders_;
  std::vector<row_t> headers_;
  std::unique_ptr<bool[]> selected_;

  std::vector<row_t> work_;
  std::unique_ptr<bool[]> isHeader_;
  std::uint32_t isHeaderCap_ = 0;
  std::vector<row_t> dummyWork_;
};

// Sort-based padding scheme:
// - Creates exactly `p=2^(r+b)` buckets, where r=NUM_RADIX_BITS and b=log2(bins).
// - Computes a public capacity U via elemPerBucket(N,p).
// - Appends exactly (p*U - N) inserted dummies whose hashKey low bits encode the
//   bucket they pad, so that later radix partitioning sees exactly U rows per
//   (partition,bin) bucket.
inline void padTableSort(table_t &tbl, std::uint32_t bins,
                         std::uint32_t numThreads) {
  PadTableSortContext ctx(bins, numThreads);
  ctx.pad(tbl);
}
