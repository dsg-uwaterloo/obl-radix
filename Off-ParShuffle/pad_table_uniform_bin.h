#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <math.h>
#include <memory>
#include <stdexcept>
#include <stdio.h>
#include <thread>
#include <vector>

#include "external/radix_partition/data-types.h"
#include "oblivious_ops.h"
#include "slice_utils.h"

constexpr double kSecParam = 64.0;

inline double lambert_w0(double x) {
  const double lower = -1.0 / M_E;
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

// Used to compute a public fixed per-bucket capacity
inline int elemPerBucket(uint64_t totalSize, uint64_t bucketCount) {
  if (bucketCount == 0 || totalSize == 0)
    return 0;
  double mu = static_cast<double>(totalSize) / static_cast<double>(bucketCount);
  if (mu <= 0.0)
    return 0;
  double alpha =
      std::log(static_cast<double>(bucketCount) * std::pow(2.0, kSecParam));
  double rhs = alpha / (M_E * mu) - (1.0 / M_E);
  rhs = std::max(rhs, -1.0 / M_E + 1e-12);
  double epsilon = std::pow(M_E, lambert_w0(rhs) + 1.0) - 1.0;
  double b = mu * (1.0 + epsilon);
  if (!std::isfinite(b) || b < 1.0)
    b = 1.0;
  return static_cast<int>(std::ceil(b));
}

namespace pad_table_uniform_bin_detail {

inline bool is_power_of_two(std::uint32_t x) {
  return x && ((x & (x - 1u)) == 0);
}

inline std::uint32_t log2_pow2(std::uint32_t x) {
  return static_cast<std::uint32_t>(__builtin_ctz(x));
}

inline std::uint32_t bool_to_u32(bool x) {
  return static_cast<std::uint32_t>(x);
}

inline std::uint32_t ct_mask_u32(std::uint32_t sel01) { return 0u - sel01; }

inline std::uint32_t ct_select_u32(std::uint32_t a, std::uint32_t b,
                                   std::uint32_t sel01) {
  const std::uint32_t mask = ct_mask_u32(sel01); // 0x00000000 or 0xffffffff
  return (a & ~mask) | (b & mask);
}

inline std::uint32_t ct_lt_u32(std::uint32_t a, std::uint32_t b) {
  // Returns 1 iff a < b, computed without branches (uses 64-bit borrow).
  return static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(a) - static_cast<std::uint64_t>(b)) >> 63);
}

inline std::uint32_t ct_le_u32(std::uint32_t a, std::uint32_t b) {
  // a <= b  <=>  !(b < a)
  return 1u ^ ct_lt_u32(b, a);
}

inline std::uint32_t mask_lowbits(std::uint32_t bits) {
  if (bits >= 32u)
    return 0xffffffffu;
  if (bits == 0u)
    return 0u;
  return (1u << bits) - 1u;
}

inline std::uint32_t bucket_tag_1based(const row_t &row,
                                       std::uint32_t lowMask) {
  // Inserted padding dummies are marked with idx==UINT32_MAX. They have t=0
  // (until after OPartition completes).
  const std::uint32_t isInsertedDummy =
      bool_to_u32(row.idx == std::numeric_limits<std::uint32_t>::max());
  const std::uint32_t t = (row.hashKey & lowMask) + 1u; // in [1..p]
  return ct_select_u32(t, 0u, isInsertedDummy);
}

struct Range {
  std::uint32_t begin;
  std::uint32_t len;
  std::uint32_t l; // bucket range [l, r)
  std::uint32_t r;
};

template <class Func>
inline void parallel_slices_limited(std::uint32_t maxThreads,
                                    const std::vector<Slice> &slices,
                                    const Func &fn) {
  if (slices.empty())
    return;

  const std::uint32_t threads = std::max<std::uint32_t>(
      1u, std::min<std::uint32_t>(maxThreads,
                                  static_cast<std::uint32_t>(slices.size())));
  if (threads == 1u) {
    fn(0u, slices[0]);
    for (std::size_t t = 1; t < slices.size(); ++t)
      fn(t, slices[t]);
    return;
  }

  std::vector<std::thread> pool;
  pool.reserve(threads - 1u);
  for (std::uint32_t t = 0; t + 1u < threads; ++t) {
    const Slice sl = slices[t];
    pool.emplace_back([&, t, sl] { fn(t, sl); });
  }
  // Run the last worker slice on the caller thread. If we built more slices
  // than `threads`, run the remaining slices sequentially on the caller thread.
  fn(threads - 1u, slices[threads - 1u]);
  for (std::size_t t = threads; t < slices.size(); ++t)
    fn(t, slices[t]);

  for (auto &th : pool)
    th.join();
}

// Oblivious partitioning (OPartition) based on Algorithms 2 & 3 from Jodes.
// It rearranges `rows[begin:begin+len)` (where len=(r-l)*U) so that buckets
// in [l,m) occupy the first (m-l)*U positions, and buckets in [m,r) occupy
// the rest. Recurses until each bucket has a contiguous block of U.
inline void opartition_rec(row_t *rows, bool *selected, const Range &rng,
                           std::uint32_t U, std::uint32_t lowMask,
                           std::uint32_t numThreads) {
  const std::uint32_t bucketCount = rng.r - rng.l;
  if (bucketCount <= 1u || rng.len <= U)
    return;

  const std::uint32_t m = (rng.l + rng.r) / 2u;
  const std::uint32_t leftBuckets = m - rng.l;
  const std::uint32_t leftTarget = leftBuckets * U;

  constexpr std::uint32_t kParallelThreshold = 1u << 12; // 4096
  const std::uint32_t scanThreads =
      (numThreads > 1 && rng.len >= kParallelThreshold)
          ? std::min(numThreads, rng.len)
          : 1u; // total threads including the caller
  const auto slices = buildSlices(rng.len, scanThreads);

  // Pass A: compute per-slice counts of real-left and dummy (t==0).
  std::vector<std::uint32_t> realLeftCount(slices.size(), 0);
  std::vector<std::uint32_t> dummyCount(slices.size(), 0);
  parallel_slices_limited(
      scanThreads, slices, [&](std::size_t sliceIndex, const Slice &sl) {
        std::uint32_t realAcc = 0;
        std::uint32_t dummyAcc = 0;
        for (std::uint32_t off = sl.begin; off < sl.end; ++off) {
          const row_t &row = rows[rng.begin + off];
          const std::uint32_t t = bucket_tag_1based(row, lowMask);
          const std::uint32_t isDummy = bool_to_u32(t == 0u);
          const std::uint32_t isRealLeft = (1u ^ isDummy) & ct_le_u32(t, m);
          realAcc += isRealLeft;
          dummyAcc += isDummy;
        }
        realLeftCount[static_cast<std::uint32_t>(sliceIndex)] = realAcc;
        dummyCount[static_cast<std::uint32_t>(sliceIndex)] = dummyAcc;
      });

  std::uint32_t totalRealLeft = 0;
  std::uint32_t totalDummy = 0;
  for (std::size_t i = 0; i < slices.size(); ++i) {
    totalRealLeft += realLeftCount[i];
    totalDummy += dummyCount[i];
  }
  if (totalRealLeft > leftTarget) {
    throw std::runtime_error(
        "padTableUniformBin: overflow (real-left > leftTarget); increase U or "
        "reduce bins");
  }
  const std::uint32_t K =
      leftTarget - totalRealLeft; // number of inserted dummies to place left

  // Pass B: compute dummy ranks (stable, by position) and build selection
  // bitmap.
  std::vector<std::uint32_t> dummyPrefix(slices.size(), 0);
  std::uint32_t run = 0;
  for (std::size_t i = 0; i < slices.size(); ++i) {
    dummyPrefix[i] = run;
    run += dummyCount[i];
  }
  (void)totalDummy; // retained for future debug checks

  parallel_slices_limited(
      scanThreads, slices, [&](std::size_t sliceIndex, const Slice &sl) {
        std::uint32_t localDummySeen = 0;
        const std::uint32_t baseRank =
            dummyPrefix[static_cast<std::uint32_t>(sliceIndex)];

        for (std::uint32_t off = sl.begin; off < sl.end; ++off) {
          const std::uint32_t i = rng.begin + off;
          const std::uint32_t t = bucket_tag_1based(rows[i], lowMask);
          const std::uint32_t isDummy = bool_to_u32(t == 0u);
          const std::uint32_t isRealLeft = (1u ^ isDummy) & ct_le_u32(t, m);

          const std::uint32_t dummyRank =
              baseRank + localDummySeen; // 0-based among dummies
          const std::uint32_t isDummyLeft = isDummy & ct_lt_u32(dummyRank, K);

          selected[i] = (isRealLeft | isDummyLeft) != 0u;
          localDummySeen += isDummy;
        }
      });

  obli_compact_rows(rows + rng.begin, selected + rng.begin, rng.len,
                    numThreads);

  const std::uint32_t rightBegin = rng.begin + leftTarget;
  const std::uint32_t rightLen = rng.len - leftTarget;
  const Range left{rng.begin, leftTarget, rng.l, m};
  const Range right{rightBegin, rightLen, m, rng.r};

  // Parallelize recursion using a fixed thread budget (public control).
  const std::uint32_t canSplit =
      bool_to_u32(numThreads > 1 && bucketCount >= 2u);
  const std::uint32_t rightThreads =
      canSplit ? std::max<std::uint32_t>(1u, numThreads / 2u) : 1u;
  const std::uint32_t leftThreads =
      canSplit ? std::max<std::uint32_t>(1u, numThreads - rightThreads) : 1u;

  std::thread rightWorker;
  if (canSplit) {
    rightWorker = std::thread([&] {
      opartition_rec(rows, selected, right, U, lowMask, rightThreads);
    });
  } else {
    opartition_rec(rows, selected, right, U, lowMask, rightThreads);
  }
  opartition_rec(rows, selected, left, U, lowMask, leftThreads);
  if (rightWorker.joinable())
    rightWorker.join();
}

inline void retag_inserted_dummies(row_t *rows, std::uint32_t n,
                                   std::uint32_t U, std::uint32_t lowMask,
                                   std::uint32_t numThreads) {
  numThreads = std::max<std::uint32_t>(1, numThreads);
  const std::uint32_t maxIdx = std::numeric_limits<std::uint32_t>::max();

  const std::uint32_t threads =
      std::max<std::uint32_t>(1u, std::min<std::uint32_t>(numThreads, n));
  const auto slices = buildSlices(n, threads);

  parallel_slices_limited(
      threads, slices, [&](std::size_t /*sliceIndex*/, const Slice &sl) {
        if (sl.begin >= sl.end)
          return;

        // Avoid division in the inner loop: bucketIndex == i / U.
        // U is public and >= 1 in this code path.
        std::uint32_t bucketIndex = sl.begin / U;
        std::uint64_t nextBoundary =
            (static_cast<std::uint64_t>(bucketIndex) + 1ull) *
            static_cast<std::uint64_t>(U);

        for (std::uint32_t i = sl.begin; i < sl.end; ++i) {
          if (static_cast<std::uint64_t>(i) == nextBoundary) {
            ++bucketIndex;
            nextBoundary += static_cast<std::uint64_t>(U);
          }

          const std::uint32_t isInserted = bool_to_u32(rows[i].idx == maxIdx);
          const std::uint32_t newHash =
              (rows[i].hashKey & ~lowMask) | (bucketIndex & lowMask);
          rows[i].hashKey = ct_select_u32(rows[i].hashKey, newHash, isInserted);
        }
      });
}

} // namespace pad_table_uniform_bin_detail

// Produces a table where every combined (partition,bin) bucket has exactly U
// rows, without sorting all elements by bucket id.
inline void padTableUniformBin(table_t &tbl, std::uint32_t bins,
                               std::uint32_t numThreads) {
  using namespace pad_table_uniform_bin_detail;
  numThreads = std::max<std::uint32_t>(1, numThreads);

  if (tbl.num_tuples == 0)
    return;

  constexpr std::uint32_t rBits = NUM_RADIX_BITS;
  const std::uint32_t bBits = log2_pow2(bins);
  const std::uint32_t totalBits = rBits + bBits;
  const std::uint32_t lowMask = mask_lowbits(totalBits);

  const std::uint64_t partitions = 1ull << rBits;
  const std::uint64_t totalBins64 =
      partitions * static_cast<std::uint64_t>(bins);

  const std::uint32_t N = static_cast<std::uint32_t>(tbl.num_tuples);
  const std::uint32_t U = static_cast<std::uint32_t>(
      std::max<int>(1, elemPerBucket(N, totalBins64)));

  // Let U * bins balls be uniformly randomly put into bins and p be the
  // probability indicating that any bin will be empty. Then, p ≤ bins * e^-U.
  const double p = (double)bins * exp(-(double)U);
  // printf("p=%.10g bins=%u U=%u\n", p, bins, U);
  if (p > 0.001) {
    fprintf(stderr, "[WARNING] Some bins may be empty. Reconfigure radix "
                    "parameters or bins.\n");
  }

  const std::uint64_t finalN64 = totalBins64 * static_cast<std::uint64_t>(U);
  if (finalN64 > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error(
        "padTableUniformBin: input too large for current oblivious compaction");
  const std::uint32_t finalN = static_cast<std::uint32_t>(finalN64);

  // Algorithm 2, line 1: extend to Up with (x=⊥, t=0). Here we mark inserted
  // dummies via idx==UINT32_MAX and t is derived on-the-fly from hashKey.
  row_t *expanded = new row_t[finalN];
  std::memcpy(expanded, tbl.tuples,
              static_cast<std::size_t>(N) * sizeof(row_t));

  row_t dummy{};
  dummy.idx = std::numeric_limits<std::uint32_t>::max();
  dummy.cntSelf = 0;
  dummy.cntExpand = 0;
  dummy.hashKey = 0;
  dummy.shuffledIdx = 0;

  for (std::uint32_t i = N; i < finalN; ++i)
    expanded[i] = dummy;

  delete[] tbl.tuples;
  tbl.tuples = expanded;
  tbl.num_tuples = finalN;

  auto selected = std::make_unique<bool[]>(finalN);
  opartition_rec(tbl.tuples, selected.get(),
                 Range{0, finalN, 0, static_cast<std::uint32_t>(totalBins64)},
                 U, lowMask, numThreads);

  // After OPartition, inserted dummies are in arbitrary buckets (t=0). Retag
  // them to the bucket (partition,bin) implied by their final position so that
  // radix partitioning / bucket indexing sees uniform per-bucket sizes.
  retag_inserted_dummies(tbl.tuples, finalN, U, lowMask, numThreads);

  // #ifndef NDEBUG
  //   {
  //     const std::uint32_t p = static_cast<std::uint32_t>(totalBins64);
  //     std::vector<std::uint32_t> counts(p, 0);
  //     for (std::uint32_t i = 0; i < finalN; ++i) {
  //       const std::uint32_t b = tbl.tuples[i].hashKey & lowMask;
  //       if (b >= p) {
  //         throw std::runtime_error("padTableUniformBin: bucket id out of
  //         range");
  //       }
  //       ++counts[b];
  //     }
  //     for (std::uint32_t b = 0; b < p; ++b) {
  //       if (counts[b] != U) {
  //         throw std::runtime_error(
  //             "padTableUniformBin: verifier failed (bucket size != U)");
  //       }
  //     }
  //   }
  // #endif
}
