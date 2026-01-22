#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <immintrin.h>
#include <inttypes.h>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <sys/random.h>
#include <sys/types.h>
#include <tbb/global_control.h>
#include <tbb/parallel_sort.h>

#include "align_table.h"
#include "backfill_dummies.h"
#include "carry_forward.h"
#include "external_memory/algorithm/kway_butterfly_sort.hpp"
#include "inputs.h"
#include "merge.h"
#include "oblivious_ops.h"
#include "parallel_counts.h"
#include "parallel_index.h"
#include "prefix_sum_expand.h"
#include "raw_vector.h"
#include "replace_dummies.h"
#include "slice_utils.h"

extern "C" {
#include "bitonic.h"
#include "radix_join_counts.h"
#include "threading.h"
}

// #define PRE_SORTED // use this if your tables are already sorted

// Global timer
std::chrono::high_resolution_clock::time_point tStart, tEnd;
// std::uint32_t SECRET;

constexpr double kSecParam = 128.0;

static void printOutput(const char *label, const table_t &tbl) {
  printf("%s (num_tuples=%" PRIu64 ")\n", label,
         static_cast<uint64_t>(tbl.num_tuples));
  for (uint64_t i = 0; i < tbl.num_tuples; ++i) {
    const row_t &rec = tbl.tuples[i];
    printf("  [%" PRIu64 "] key=%u cntSelf=%u cntExpand=%u idx=%u hashKey=%u\n",
           i, rec.key, rec.cntSelf, rec.cntExpand, rec.idx, rec.hashKey);
  }
}

static std::ofstream &padDebugStream() {
  static std::ofstream out("pad_debug.log", std::ios::out | std::ios::trunc);
  return out;
}

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

inline std::uint32_t prev_power_of_two(std::uint32_t n) {
  if (n <= 1)
    return 1;
  const unsigned leading = static_cast<unsigned>(__builtin_clz(n));
  return 1u << (31u - leading);
}

/**
 * Find the maximum number of bins that achieves a target probability
 * Lemma 1: m * exp(-n/m) ≈ target_p
 */
inline std::pair<std::uint32_t, double>
findMaxBins(double n, double target_p = 0.001, double eps = 1e-6) {
  // return {1, 0.001};
  int i;
  double low = 1, high = n, m = 0, p = 0;
  for (i = 0; i < 100; ++i) {
    m = (low + high) / 2.0;
    p = m * std::exp(-n / m);
    if (std::fabs(p - target_p) < eps)
      break;
    (p > target_p) ? (high = m) : (low = m);
  }

  if (i == 100) {
    std::cerr << "[WARNING] Lemma 1 unsatisfied. Reconfigure radix parameters."
              << std::endl;
  }

  return {prev_power_of_two(static_cast<std::uint32_t>(std::ceil(m))), p};
}

static void shuffleTable(table_t &tbl, std::uint32_t numThreads) {
  if (tbl.num_tuples == 0)
    return;
  RawVector<row_t> vec{tbl.tuples, static_cast<std::size_t>(tbl.num_tuples)};
  try {
    if (vec.size() <= 512) {
      ::StdVector<row_t> tmp(vec.size());
      std::copy(vec.begin(), vec.end(), tmp.begin());
      EM::Algorithm::OrShuffle(tmp);
      std::copy(tmp.begin(), tmp.end(), vec.begin());
      return;
    }
    // Provide a heap size proportional to the data; back off on bad_alloc.
    constexpr uint64_t MAX_HEAP = 190ULL << 30;
    constexpr uint64_t MIN_HEAP = 64ULL << 20;
    uint64_t heapSize = std::max<uint64_t>(vec.size(), 4096UL) *
                        sizeof(EM::Algorithm::TaggedT<row_t>) * 2;
    heapSize = std::max<uint64_t>(heapSize, MIN_HEAP);
    heapSize = std::min<uint64_t>(heapSize, MAX_HEAP);

    while (true) {
      try {
        EM::Algorithm::KWayButterflyOShuffleFixedThreads(vec.begin(), vec.end(),
                                                         0, heapSize,
                                                         (int)numThreads);
        break;
      } catch (const std::bad_alloc &) {
        if (heapSize <= MIN_HEAP) {
          throw;
        }
        heapSize /= 2;
      }
    }
  } catch (const std::exception &e) {
    fprintf(stderr,
            "Shuffle failed (%s) at size %zu; using StdVector OrShuffle "
            "fallback\n",
            e.what(), static_cast<size_t>(tbl.num_tuples));
    ::StdVector<row_t> tmp(vec.size());
    std::copy(vec.begin(), vec.end(), tmp.begin());
    EM::Algorithm::OrShuffle(tmp);
    std::copy(tmp.begin(), tmp.end(), vec.begin());
  }
}

// Synthesizes a dummy row that hashes into a specific (partition, bin) pair.
inline row_t make_dummy_row(uint32_t partitionId, uint32_t binId,
                            uint32_t binsPerPartition) {
  row_t dummy{};
  // dummy.key = triple32(partitionId ^ (binId * 2654435761u));
  // dummy.key = UINT32_MAX;
  dummy.cntSelf = 0;
  dummy.cntExpand = 0;
  // dummy.idx = UINT32_MAX;
  dummy.hashKey = (binId << NUM_RADIX_BITS) | partitionId;
  // std::memset(dummy.pay, 0, DATA_LENGTH);
  return dummy;
}

// Pads a table so every radix partition/bin pair has identical capacity
static void padTableUniform(table_t &tbl, uint32_t binsPerPartition,
                            uint32_t numThreads) {
  // Number of top-level partitions (radix buckets for the high NUM_RADIX_BITS)
  const uint32_t partitions = 1u << NUM_RADIX_BITS;
  if (tbl.num_tuples == 0 || partitions == 0 || binsPerPartition == 0)
    return;

  // Total number of (partition, bin) pairs we must pad to a uniform size.
  const uint64_t totalBins =
      static_cast<uint64_t>(partitions) * binsPerPartition;
  if (totalBins == 0)
    return;

  // Public cap from the analytical bound; ensures per-bin load meets the
  // oblivious spill requirements even before we inspect actual data.
  const uint32_t initialPerBin = static_cast<uint32_t>(
      std::max(1, elemPerBucket(tbl.num_tuples, totalBins)));
  if (initialPerBin == 0)
    return;

  const uint32_t partitionMask = partitions - 1u;
  // For binsPerPartition that are powers of two we can use bit masks instead of
  // slow modulo operations when extracting bin IDs from the hash.
  const bool binsPowerOfTwo = (binsPerPartition & (binsPerPartition - 1u)) == 0;
  const uint32_t binMask = binsPowerOfTwo ? (binsPerPartition - 1u) : 0;
  // Break the input into slices so multiple threads can count their own
  // histograms without sharing state or locks.
  const uint32_t countingThreads =
      std::max<uint32_t>(1, std::min<uint32_t>(numThreads, tbl.num_tuples));
  auto countSlices =
      buildSlices(static_cast<uint32_t>(tbl.num_tuples), countingThreads);
  if (countSlices.empty())
    return;

  // Guard against overflow when allocating one histogram per slice.
  if (totalBins > std::numeric_limits<size_t>::max() /
                      std::max<size_t>(size_t{1}, countSlices.size())) {
    throw std::bad_alloc();
  }
  // Each slice keeps a private histogram to avoid shared writes; we'll reduce
  // them after the parallel pass.
  const size_t histogramSize =
      countSlices.size() * static_cast<size_t>(totalBins);
  std::vector<uint32_t> localCounts(histogramSize, 0);

  // Worker that counts how many rows within its slice fall into each
  // (partition, bin) bucket. Uses AVX2 gather when the bin count is a power of
  // two so we can process 8 rows per iteration without leaking branch
  // decisions.
  std::vector<uint32_t> binIndexTable(totalBins);
  for (uint64_t idx = 0; idx < totalBins; ++idx)
    binIndexTable[idx] = static_cast<uint32_t>(idx);

  auto countWorker = [&](size_t tid) {
    const Slice sl = countSlices[tid];
    const row_t *rows = tbl.tuples + sl.begin;
    const size_t len = sl.end - sl.begin;
    uint32_t *local = localCounts.data() + tid * totalBins;
    constexpr size_t kBlockSize = 64;
    std::array<uint32_t, kBlockSize> blockBins{};
    for (size_t blockStart = 0; blockStart < len; blockStart += kBlockSize) {
      const size_t blockLen =
          std::min(kBlockSize, static_cast<size_t>(len - blockStart));
      for (size_t r = 0; r < blockLen; ++r) {
        const row_t &row = rows[blockStart + r];
        const uint32_t partId = row.hashKey & partitionMask;
        uint32_t binId;
        if (binsPowerOfTwo) {
          binId = (row.hashKey >> NUM_RADIX_BITS) & binMask;
        } else {
          binId = (row.hashKey >> NUM_RADIX_BITS) % binsPerPartition;
        }
        blockBins[r] = static_cast<uint32_t>(
            static_cast<uint64_t>(partId) * binsPerPartition + binId);
      }
#if defined(__AVX2__)
      const __m256i oneVec = _mm256_set1_epi32(1);
      uint64_t idx = 0;
      for (; idx + 8 <= totalBins; idx += 8) {
        const __m256i ids = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(binIndexTable.data() + idx));
        __m256i acc = _mm256_setzero_si256();
        for (size_t r = 0; r < blockLen; ++r) {
          const __m256i target =
              _mm256_set1_epi32(static_cast<int>(blockBins[r]));
          const __m256i mask = _mm256_cmpeq_epi32(ids, target);
          acc = _mm256_add_epi32(acc, _mm256_and_si256(mask, oneVec));
        }
        __m256i counts =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(local + idx));
        counts = _mm256_add_epi32(counts, acc);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(local + idx), counts);
      }
      for (; idx < totalBins; ++idx) {
        uint32_t acc = 0;
        for (size_t r = 0; r < blockLen; ++r)
          acc += (blockBins[r] == binIndexTable[idx]);
        local[idx] += acc;
      }
#else
      for (uint64_t idx = 0; idx < totalBins; ++idx) {
        uint32_t acc = 0;
        for (size_t r = 0; r < blockLen; ++r)
          acc += (blockBins[r] == binIndexTable[idx]);
        local[idx] += acc;
      }
#endif
    }
  };

  // Launch threads (if needed) so each slice fills its histogram, then join.
  if (countSlices.size() == 1) {
    countWorker(0);
  } else {
    std::vector<std::thread> pool;
    pool.reserve(countSlices.size() - 1);
    for (size_t tid = 1; tid < countSlices.size(); ++tid)
      pool.emplace_back(countWorker, tid);
    countWorker(0);
    for (auto &th : pool)
      th.join();
  }

  // Combine per-slice histograms in parallel so each bin sum is independent.
  std::vector<uint64_t> binCounts(totalBins, 0);
  const uint32_t binThreadCount =
      std::max<uint32_t>(1, std::min<uint32_t>(numThreads, totalBins));
  auto binSlices =
      buildSlices(static_cast<uint32_t>(totalBins), binThreadCount);
  auto reduceWorker = [&](size_t tid) {
    const Slice sl = binSlices[tid];
    for (uint32_t b = sl.begin; b < sl.end; ++b) {
      uint64_t sum = 0;
      size_t offset = b;
      for (size_t slice = 0; slice < countSlices.size(); ++slice) {
        sum += localCounts[offset];
        offset += totalBins;
      }
      binCounts[b] = sum;
    }
  };
  if (binSlices.size() == 1) {
    reduceWorker(0);
  } else {
    std::vector<std::thread> pool;
    pool.reserve(binSlices.size() - 1);
    for (size_t tid = 1; tid < binSlices.size(); ++tid)
      pool.emplace_back(reduceWorker, tid);
    reduceWorker(0);
    for (auto &th : pool)
      th.join();
  }


  // Final per-bin capacity: maximum of the analytical bound and the maximum
  // load we observed. This guarantees we neither shrink real data nor reveal
  // which bins were heavy.
  // const uint32_t perBin =
  //     static_cast<uint32_t>(std::max<uint64_t>(initialPerBin, maxBinLoad));
  // const uint64_t targetSize = totalBins * static_cast<uint64_t>(perBin);
  const uint32_t perBin = initialPerBin; // REMOVEEEEE
  const uint64_t targetSize = totalBins * static_cast<uint64_t>(initialPerBin);
  // {
  //   auto &dbg = padDebugStream();
  //   dbg << "[PAD] binsPerPartition=" << binsPerPartition
  //       << " totalBins=" << totalBins << " perBin=" << perBin
  //       << " current=" << tbl.num_tuples << " target=" << targetSize << '\n';
  // }
  if (tbl.num_tuples >= targetSize)
    return;

  // Allocate a new buffer sized exactly for the padded layout and copy over the
  // real rows. The trailing portion will be filled with dummy rows next.
  row_t *expanded = new row_t[targetSize];
  if (tbl.num_tuples <= static_cast<uint32_t>(numThreads)) {
    std::memcpy(expanded, tbl.tuples,
                static_cast<size_t>(tbl.num_tuples) * sizeof(row_t));
  } else {
    auto copySlices = buildSlices(
        tbl.num_tuples,
        std::max<uint32_t>(1, std::min<uint32_t>(numThreads, tbl.num_tuples)));
    auto copyWorker = [&](size_t tid) {
      const Slice sl = copySlices[tid];
      std::memcpy(expanded + sl.begin, tbl.tuples + sl.begin,
                  static_cast<size_t>(sl.end - sl.begin) * sizeof(row_t));
    };
    std::vector<std::thread> pool;
    pool.reserve(copySlices.size() > 0 ? copySlices.size() - 1 : 0);
    for (size_t tid = 1; tid < copySlices.size(); ++tid)
      pool.emplace_back(copyWorker, tid);
    copyWorker(0);
    for (auto &th : pool)
      th.join();
  }

  //   // For each bin remember how many dummies we still need and where their
  //   block
  //   // begins inside the expanded array.
  //   std::vector<uint32_t> deficits(totalBins);
  //   auto deficitWorker = [&](size_t tid) {
  //     const Slice sl = binSlices[tid];
  //     for (uint32_t idx = sl.begin; idx < sl.end; ++idx) {
  //       const uint32_t capped =
  //           static_cast<uint32_t>(std::min<uint64_t>(binCounts[idx],
  //           perBin));
  //       const uint32_t deficit = perBin - capped;
  //       deficits[idx] = deficit;
  //     }
  //   };
  //   if (binSlices.size() == 1) {
  //     deficitWorker(0);
  //   } else {
  //     std::vector<std::thread> pool;
  //     pool.reserve(binSlices.size() - 1);
  //     for (size_t tid = 1; tid < binSlices.size(); ++tid)
  //       pool.emplace_back(deficitWorker, tid);
  //     deficitWorker(0);
  //     for (auto &th : pool)
  //       th.join();
  //   }

  //   const uint64_t padStart = tbl.num_tuples;
  //   for (uint64_t padIdx = padStart; padIdx < targetSize; ++padIdx) {
  //     uint32_t chosenBin = 0;
  //     uint32_t alreadySelected = 0;
  // #if defined(__AVX2__)
  //     const __m256i zeroVec = _mm256_setzero_si256();
  //     const __m256i oneVec = _mm256_set1_epi32(1);
  //     uint64_t idx = 0;
  //     for (; idx + 8 <= totalBins; idx += 8) {
  //       __m256i defs = _mm256_loadu_si256(
  //           reinterpret_cast<const __m256i *>(deficits.data() + idx));
  //       __m256i hasDef = _mm256_cmpgt_epi32(defs, zeroVec);
  //       alignas(32) uint32_t maskBuf[8];
  //       _mm256_store_si256(reinterpret_cast<__m256i *>(maskBuf), hasDef);
  //       for (int lane = 0; lane < 8; ++lane) {
  //         const uint32_t b = static_cast<uint32_t>(idx + lane);
  //         const uint32_t hasDeficit =
  //             static_cast<uint32_t>(-static_cast<int32_t>(maskBuf[lane] !=
  //             0));
  //         const uint32_t select = hasDeficit & (~alreadySelected);
  //         chosenBin = (select & b) | (~select & chosenBin);
  //         alreadySelected |= hasDeficit;
  //       }
  //     }
  //     for (; idx < totalBins; ++idx) {
  //       const uint32_t hasDeficit =
  //           static_cast<uint32_t>(-static_cast<int32_t>(deficits[idx] != 0));
  //       const uint32_t select = hasDeficit & (~alreadySelected);
  //       chosenBin = (select & static_cast<uint32_t>(idx)) |
  //                   (~select & chosenBin);
  //       alreadySelected |= hasDeficit;
  //     }
  // #else
  //     for (uint32_t b = 0; b < totalBins; ++b) {
  //       const uint32_t hasDeficit =
  //           static_cast<uint32_t>(-static_cast<int32_t>(deficits[b] != 0));
  //       const uint32_t select = hasDeficit & (~alreadySelected);
  //       chosenBin = (select & b) | (~select & chosenBin);
  //       alreadySelected |= hasDeficit;
  //     }
  // #endif
  //     const uint32_t partId = chosenBin / binsPerPartition;
  //     const uint32_t binId = chosenBin - partId * binsPerPartition;
  //     row_t dummy = make_dummy_row(partId, binId, binsPerPartition);
  //     dummy.idx = static_cast<uint32_t>(padIdx);
  //     expanded[padIdx] = dummy;
  // #if defined(__AVX2__)
  //     idx = 0;
  //     const __m256i chosenVec =
  //     _mm256_set1_epi32(static_cast<int>(chosenBin)); for (; idx + 8 <=
  //     totalBins; idx += 8) {
  //       __m256i ids = _mm256_loadu_si256(
  //           reinterpret_cast<const __m256i *>(binIndexTable.data() + idx));
  //       __m256i match = _mm256_cmpeq_epi32(ids, chosenVec);
  //       __m256i defs = _mm256_loadu_si256(
  //           reinterpret_cast<const __m256i *>(deficits.data() + idx));
  //       defs = _mm256_sub_epi32(defs, _mm256_and_si256(match, oneVec));
  //       _mm256_storeu_si256(reinterpret_cast<__m256i *>(deficits.data() +
  //       idx),
  //                           defs);
  //     }
  //     for (; idx < totalBins; ++idx) {
  //       const uint32_t isChosen =
  //           static_cast<uint32_t>(-static_cast<int32_t>(binIndexTable[idx] ==
  //                                                       chosenBin));
  //       deficits[idx] -= (isChosen & 1u);
  //     }
  // #else
  //     for (uint32_t b = 0; b < totalBins; ++b) {
  //       const uint32_t isChosen =
  //           static_cast<uint32_t>(-static_cast<int32_t>(binIndexTable[b] ==
  //                                                       chosenBin));
  //       deficits[b] -= (isChosen & 1u);
  //     }
  // #endif
  //   }

  //   delete[] tbl.tuples;
  //   tbl.tuples = expanded;
  //   tbl.num_tuples = targetSize;
}

static std::string toBinary(uint32_t value) {
  std::string out(32, '0');
  for (int i = 31; i >= 0; --i) {
    out[31 - i] = ((value >> i) & 1u) ? '1' : '0';
  }
  return out;
}

static void dumpTableDebug(const char *label, const table_t &tbl,
                           uint32_t binsPerPartition) {
  const uint32_t partMask = (1u << NUM_RADIX_BITS) - 1u;
  const bool binsPowerTwo = (binsPerPartition != 0) &&
                            ((binsPerPartition & (binsPerPartition - 1u)) == 0);
  const uint32_t binMask = binsPowerTwo ? (binsPerPartition - 1u) : 0u;
  const uint32_t partitions = (partMask + 1u);
  const uint32_t totalBins =
      (binsPerPartition == 0) ? partitions : partitions * binsPerPartition;
  std::vector<uint64_t> perPart(partitions, 0);
  std::vector<uint64_t> perBin(totalBins, 0);
  auto &dbg = padDebugStream();
  dbg << "[PAD-DUMP] Table=" << label << " rows=" << tbl.num_tuples << '\n';
  for (uint32_t i = 0; i < tbl.num_tuples; ++i) {
    const row_t &row = tbl.tuples[i];
    const uint32_t partId = row.hashKey & partMask;
    uint32_t binId;
    if (binsPerPartition == 0)
      binId = 0;
    else if (binsPowerTwo)
      binId = (row.hashKey >> NUM_RADIX_BITS) & binMask;
    else
      binId = (row.hashKey >> NUM_RADIX_BITS) % binsPerPartition;
    ++perPart[partId];
    const uint32_t binIdx =
        (binsPerPartition == 0) ? partId : partId * binsPerPartition + binId;
    ++perBin[binIdx];
    const std::string binStr = toBinary(row.key);
    dbg << "[PAD-DUMP] " << label << '[' << i << "] key=" << row.key
        << " key_bin=" << binStr << " part=" << partId << " bin=" << binId
        << '\n';
  }
  dbg << "[PAD-DUMP] Summary partitions";
  for (uint32_t p = 0; p < partitions; ++p)
    dbg << " p" << p << '=' << perPart[p];
  dbg << '\n';
  dbg << "[PAD-DUMP] Summary bins";
  for (uint32_t p = 0; p < partitions; ++p) {
    dbg << " [part " << p << ':';
    for (uint32_t b = 0; b < binsPerPartition; ++b) {
      const uint32_t idx =
          (binsPerPartition == 0) ? p : p * binsPerPartition + b;
      dbg << " bin" << b << '=' << perBin[idx];
    }
    dbg << ']';
  }
  dbg << '\n';
  dbg.flush();
}

inline __m128i generateSecret() {
  alignas(16) __m128i secret; // 128-bit aligned buffer
  ssize_t ret = getrandom(&secret, sizeof(secret), 0);
  if (ret != sizeof(secret))
    throw std::runtime_error("getrandom failed");
  return secret;
}

int main(int argc, char *argv[]) {
  printf("[INFO] Set number of radix bits and passes in the top-level "
         "CMakeLists.txt.\n");
  std::uint32_t numThreads = 32;
  std::string inputPath = "../../datasets/real/amazon.txt";

  if (argc > 1)
    numThreads = std::max<std::uint32_t>(1, std::stoul(argv[1]));
  if (argc > 2)
    inputPath = argv[2];
  if (argc > 3) {
    std::cerr << "Program takes 2 arguments: number of threads and input "
                 "filepath."
              << std::endl;
    return 1;
  }
  printf("Input: %s\n", inputPath.c_str());
  printf("Threads: %u\n", numThreads);

  // {
  //   ssize_t tmp = getrandom(&SECRET, sizeof(SECRET), 0);
  //   if (tmp != sizeof(SECRET)) {
  //     perror("Secret generation failed");
  //     exit(1);
  //   }
  // }
  __m128i SECRET = generateSecret();

  std::vector<Record> t0, t1;
  if (!load_two_tables(inputPath, t0, t1))
    exit(1);

  if (t0.size() > t1.size())
    std::swap(t0, t1);

  std::vector<Record> partR;
  partR.reserve(t0.size());
  std::vector<Record> partS;
  partS.reserve(t1.size());

  std::uint32_t thrR = std::max<std::uint32_t>(
      1, ceil((static_cast<double>(t0.size()) / (t0.size() + t1.size())) *
              numThreads));
  std::uint32_t thrS = std::max<std::uint32_t>(1, numThreads - thrR);
  printf("threads_R: %u, threads_S: %u\n", thrR, thrS);

  table_t R, S;
  R.tuples = new row_t[t0.size()];
  std::memcpy(R.tuples, t0.data(), t0.size() * sizeof(Record));
  R.num_tuples = static_cast<uint32_t>(t0.size());

  S.tuples = new row_t[t1.size()];
  std::memcpy(S.tuples, t1.data(), t1.size() * sizeof(Record));
  S.num_tuples = static_cast<uint32_t>(t1.size());
  const uint32_t originalRSize = R.num_tuples;
  const uint32_t originalSSize = S.num_tuples;

  t0.clear();
  t0.shrink_to_fit();
  t1.clear();
  t1.shrink_to_fit();

  auto slices_R = buildSlices(R.num_tuples, thrR);
  auto slices_S = buildSlices(S.num_tuples, thrS);

  tbb::global_control c(tbb::global_control::max_allowed_parallelism,
                        numThreads);

  auto padTableToSize = [&](table_t &tbl, uint32_t target) {
    if (tbl.num_tuples == target)
      return;

    row_t *expanded = new row_t[target];
    const uint32_t copyCount = std::min<uint32_t>(tbl.num_tuples, target);
    std::memcpy(expanded, tbl.tuples, copyCount * sizeof(row_t));

    if (copyCount < target) {
      row_t dummy{};
      dummy.idx = UINT32_MAX;
      dummy.cntExpand = 0;
      const uint32_t fillCount = target - copyCount;
      const uint32_t fillThreads =
          std::min<std::uint32_t>(numThreads, fillCount);
      if (fillThreads <= 1) {
        std::fill(expanded + copyCount, expanded + target, dummy);
      } else {
        auto fillSlices = buildSlices(fillCount, fillThreads);
        std::vector<std::thread> pool;
        pool.reserve(fillSlices.size());
        for (const Slice &sl : fillSlices) {
          pool.emplace_back([&, sl] {
            std::fill(expanded + copyCount + sl.begin,
                      expanded + copyCount + sl.end, dummy);
          });
        }
        for (auto &th : pool) {
          th.join();
        }
      }
    }

    delete[] tbl.tuples;
    tbl.tuples = expanded;
    tbl.num_tuples = target;
  };

  std::uint32_t m;

  printf("\nRadix bits: %u, Passes: %u\n", NUM_RADIX_BITS, NUM_PASSES);
  auto [bins, p] = findMaxBins(R.num_tuples / std::pow(2, NUM_RADIX_BITS));
  printf("Bins: %u, Lemma 1 p: %.4f\n", bins, p);

#ifndef PRE_SORTED
  total_num_threads = numThreads;
  thread_system_init();

  std::vector<std::thread> pool;
  for (size_t i = 1; i < numThreads; ++i)
    pool.emplace_back(thread_start_work);

  tStart = std::chrono::high_resolution_clock::now();
  bitonic_sort_(R.tuples, true, 0, R.num_tuples, numThreads, false);
  bitonic_sort_(S.tuples, true, 0, S.num_tuples, numThreads, false);

  thread_release_all();
  for (auto &t : pool)
    t.join();
  thread_system_cleanup();

  std::chrono::high_resolution_clock::time_point sortEnd =
      std::chrono::high_resolution_clock::now();
#else
  tStart = std::chrono::high_resolution_clock::now();
#endif

  std::thread partitionR([&] {
    std::vector<int> lastLen(slices_R.size()), mergeVal(slices_R.size() - 1);
    parallelCounts(R, slices_R, lastLen, mergeVal);
    replaceWithDummiesParallel(R, slices_R, SECRET);
    // padRStart = std::chrono::high_resolution_clock::now();
    // padTableUniform(R, bins, thrR);
    // padREnd = std::chrono::high_resolution_clock::now();
  });
  std::thread partitionS([&] {
    std::vector<int> lastLen(slices_S.size()), mergeVal(slices_S.size() - 1);
    parallelCounts(S, slices_S, lastLen, mergeVal);
    replaceWithDummiesParallel(S, slices_S, SECRET);
    // padSStart = std::chrono::high_resolution_clock::now();
    // padTableUniform(S, bins, thrS);
    // padSEnd = std::chrono::high_resolution_clock::now();
  });
  partitionR.join();
  partitionS.join();

  std::chrono::high_resolution_clock::time_point padStart =
      std::chrono::high_resolution_clock::now();
  padTableUniform(R, bins, numThreads);
  padTableUniform(S, bins, numThreads);
  std::chrono::high_resolution_clock::time_point padEnd =
      std::chrono::high_resolution_clock::now();

#ifndef PRE_SORTED
  double sortSec = std::chrono::duration_cast<std::chrono::duration<double>>(
                       sortEnd - tStart)
                       .count();
  printf("\nSorting took %f s\n", sortSec);
#endif

  double paddingSec = std::chrono::duration_cast<std::chrono::duration<double>>(
                          padEnd - padStart)
                          .count();
  printf("\nPadding took %f s\n", paddingSec);

  //   // After padTableUniform for both R and S: REMOVEE
  // for (uint32_t i = 0; i < R.num_tuples; ++i) {
  //   R.tuples[i].idx = i;
  // }
  // for (uint32_t i = 0; i < S.num_tuples; ++i) {
  //   S.tuples[i].idx = i;
  // }
  // dumpTableDebug("R", R, bins);
  // dumpTableDebug("S", S, bins);

  // uint32_t expandedRSize = R.num_tuples;
  // uint32_t expandedSSize = S.num_tuples;

  printf("Padded R size = %u\n", R.num_tuples);
  printf("Padded S size = %u\n", S.num_tuples);

  // thrR = std::max<std::uint32_t>(
  //     1, static_cast<std::uint32_t>(std::ceil(
  //            (static_cast<double>(R.num_tuples) / totalTuples) *
  //            numThreads)));
  // thrS = std::max<std::uint32_t>(1, numThreads - thrR);
  // slices_R = buildSlices(R.num_tuples, thrR);
  // slices_S = buildSlices(S.num_tuples, thrS);
/*
  std::chrono::high_resolution_clock::time_point tShuffleStart =
      std::chrono::high_resolution_clock::now();
  shuffleTable(R, numThreads);
  assign_indices_parallel(R, numThreads);
  shuffleTable(S, numThreads);
  assign_indices_parallel(S, numThreads);
  std::chrono::high_resolution_clock::time_point tShuffleEnd =
      std::chrono::high_resolution_clock::now();

  std::chrono::high_resolution_clock::time_point onStart =
      std::chrono::high_resolution_clock::now();
  RHO(&R, &S, numThreads, bins);
  std::chrono::high_resolution_clock::time_point exchangeEnd =
      std::chrono::high_resolution_clock::now();

  auto cmp = [](const row_t &a, const row_t &b) { return a.idx < b.idx; };
  tbb::parallel_sort(R.tuples, R.tuples + R.num_tuples, cmp);
  tbb::parallel_sort(S.tuples, S.tuples + S.num_tuples, cmp);

  // printOutput("R after shrinkTable", R);
  // printOutput("S after shrinkTable", S);

  // auto shrinkTable = [](table_t &tbl, uint32_t target) {
  //   if (tbl.num_tuples > target)
  //     tbl.num_tuples = target;
  // };
  // shrinkTable(R, originalRSize);
  // shrinkTable(S, originalSSize);
  R.num_tuples = originalRSize;
  S.num_tuples = originalSSize;

  // printOutput("R after shrinkTable", R);
  // printOutput("S after shrinkTable", S);

  std::thread processR([&] {
    backfillDummiesParallel(R, slices_R);
    auto selected = std::make_unique<bool[]>(R.num_tuples);
    m = prefixSumExpandParallel(R, slices_R, selected.get());
    obli_compact_rows(R.tuples, selected.get(), R.num_tuples, thrR);
    // printOutput("R after compact", R);
    padTableToSize(R, m);
    obli_distribute_rows(R.tuples, m, numThreads / 2);
    carryForwardParallel(R, buildSlices(m, numThreads / 2));
  });
  std::thread processS([&] {
    backfillDummiesParallel(S, slices_S);
    auto selected = std::make_unique<bool[]>(S.num_tuples);
    m = prefixSumExpandParallel(S, slices_S, selected.get());
    obli_compact_rows(S.tuples, selected.get(), S.num_tuples, thrS);
    // printOutput("S after obli_compact_rows", S);
    padTableToSize(S, m);
    // printOutput("S after padTableToSize", S);
    obli_distribute_rows(S.tuples, m, numThreads / 2);
    // printOutput("S after distribute_rows", S);
    carryForwardParallel(S, buildSlices(m, numThreads / 2));
  });
  processR.join();
  processS.join();

  alignTableParallel(S, buildSlices(m, numThreads), numThreads);
  std::vector<JoinRec> joinResults;
  mergeExpandedParallel(R, S, numThreads, joinResults);

  // printf("Padded R size = %u\n", expandedRSize);
  // printf("Padded S size = %u\n", expandedSSize);

  double sec =
      std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart)
          .count();
  printf("\nJoin completed in %f s\n", sec);

  double onSec =
      std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - onStart)
          .count();
  printf("\nOnline: %f s\n", onSec);
  printf("\nOffline: %f s\n", sec - onSec);

  // #ifndef PRE_SORTED
  //   double sortSec =
  //   std::chrono::duration_cast<std::chrono::duration<double>>(
  //                        sortEnd - tStart)
  //                        .count();
  //   printf("\nSorting took %f s (%.2f%% of total execution time)\n", sortSec,
  //          (sortSec * 100.0 / sec));
  // #endif

  //   double dedupSec =
  //   std::chrono::duration_cast<std::chrono::duration<double>>(
  //                         (padSStart - dedupSStart) + (padRStart -
  //                         dedupRStart)) .count();
  //   printf("\nDedup took %f s (%.2f%% of total execution time)\n", dedupSec,
  //          (dedupSec * 100.0 / sec));

  //   double paddingSec =
  //   std::chrono::duration_cast<std::chrono::duration<double>>(
  //                           (padSEnd - padSStart) + (padREnd - padRStart))
  //                           .count();
  //   printf("\nPadding took %f s (%.2f%% of total execution time)\n",
  //   paddingSec,
  //          (paddingSec * 100.0 / sec));

  double shuffleSec = std::chrono::duration_cast<std::chrono::duration<double>>(
                          tShuffleEnd - tShuffleStart)
                          .count();
  printf("\nOShuffle took %f s (%.2f%% of total execution time)\n", shuffleSec,
         (shuffleSec * 100.0 / sec));

  double exchangeSec =
      std::chrono::duration_cast<std::chrono::duration<double>>(exchangeEnd -
                                                                onStart)
          .count();
  printf("\nPartitioning and exchanging counts took %f s (%.2f%% of total "
         "execution time)\n",
         exchangeSec, (exchangeSec * 100.0 / sec));

  {
    std::ofstream outER("join.txt");
    for (const auto &j : joinResults)
      outER << j.keyR << ' ' << j.payR << ' ' << j.keyS << ' ' << j.payS
            << '\n';
  }
  printf("Join result rows: %d (written to join.txt)\n", m);
*/
  return 0;
}
