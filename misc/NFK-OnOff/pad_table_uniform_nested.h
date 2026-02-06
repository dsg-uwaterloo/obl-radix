#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "external/radix_partition/data-types.h"
#include "carry_forward.h"
#include "oblivious_ops.h"
#include "slice_utils.h"

// Implemented in `main_nested.cpp` / `main.cpp`. Used to compute a public fixed
// per-bucket capacity.
inline int elemPerBucket(uint64_t totalSize, uint64_t bucketCount);

namespace pad_table_uniform_nested_detail {

inline std::uint32_t bool_to_u32(bool x) { return static_cast<std::uint32_t>(x); }

inline std::uint32_t ct_mask_u32(std::uint32_t sel01) { return 0u - sel01; }

inline std::uint32_t ct_select_u32(std::uint32_t a, std::uint32_t b,
                                   std::uint32_t sel01) {
  const std::uint32_t mask = ct_mask_u32(sel01);
  return (a & ~mask) | (b & mask);
}

inline std::uint32_t ct_lt_u32(std::uint32_t a, std::uint32_t b) {
  return static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(a) - static_cast<std::uint64_t>(b)) >> 63);
}

inline std::uint32_t ct_le_u32(std::uint32_t a, std::uint32_t b) {
  return 1u ^ ct_lt_u32(b, a);
}

inline row_t make_dummy_row(std::uint32_t partitionId, std::uint32_t binId,
                            std::uint32_t /*binsPerPartition*/) {
  row_t dummy{};
  dummy.cntSelf = 0;
  dummy.cntExpand = 0;
  dummy.hashKey = (binId << NUM_RADIX_BITS) | partitionId;
  dummy.idx = std::numeric_limits<std::uint32_t>::max();
  dummy.shuffledIdx = 0;
  std::memset(dummy.pay, 0, sizeof(dummy.pay));
  return dummy;
}

template <class Func>
inline void parallel_slices(const std::vector<Slice> &slices, const Func &fn) {
  if (slices.empty())
    return;
  if (slices.size() == 1) {
    fn(0, slices[0]);
    return;
  }
  std::vector<std::thread> pool;
  pool.reserve(slices.size() - 1);
  for (std::size_t t = 1; t < slices.size(); ++t) {
    const Slice sl = slices[t];
    pool.emplace_back([&, t, sl] { fn(t, sl); });
  }
  fn(0, slices[0]);
  for (auto &th : pool)
    th.join();
}

} // namespace pad_table_uniform_nested_detail

// Pads a table so every radix partition/bin pair has identical capacity.
//
// This is a "nested" (histogram + deficit) variant:
// - It counts tuples per (partition,bin) using an oblivious compare-based
//   histogram (no secret-dependent indexing).
// - It computes per-bin dummy deficits (secret), prefix-sums them (secret),
//   then uses oblivious distribution + a forward scan to materialize exactly
//   `totalDummies` dummies whose low bits map to the right bins.
// - Finally, it appends those dummies to the original table.
inline void padTableUniform(table_t &tbl, std::uint32_t binsPerPartition,
                            std::uint32_t numThreads) {
  using namespace pad_table_uniform_nested_detail;

  const std::uint32_t partitions = 1u << NUM_RADIX_BITS;
  if (tbl.num_tuples == 0 || partitions == 0 || binsPerPartition == 0)
    return;

  const std::uint64_t totalBins =
      static_cast<std::uint64_t>(partitions) * static_cast<std::uint64_t>(binsPerPartition);
  if (totalBins == 0)
    return;
  if (totalBins > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw std::runtime_error("padTableUniformNested: totalBins too large for 32-bit indexing");
  }

  numThreads = std::max<std::uint32_t>(1, numThreads);
  const std::uint32_t N = static_cast<std::uint32_t>(tbl.num_tuples);
  const std::uint32_t perBin = static_cast<std::uint32_t>(
      std::max<int>(1, elemPerBucket(tbl.num_tuples, totalBins)));
  if (perBin == 0)
    return;

  const std::uint64_t targetSize64 = totalBins * static_cast<std::uint64_t>(perBin);
  if (targetSize64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw std::runtime_error("padTableUniformNested: target size too large for 32-bit indexing");
  }
  const std::uint32_t targetSize = static_cast<std::uint32_t>(targetSize64);
  if (tbl.num_tuples >= targetSize)
    return;
  const std::uint32_t totalDummies = targetSize - N; // public
  if (totalDummies == 0)
    return;

  const std::uint32_t partitionMask = partitions - 1u;
  const bool binsPowerOfTwo = (binsPerPartition & (binsPerPartition - 1u)) == 0;
  const std::uint32_t binMask = binsPowerOfTwo ? (binsPerPartition - 1u) : 0u;

  // ---- Oblivious histogram count of tuples per bin (secret outputs) ----
  const std::uint32_t countingThreads =
      std::max<std::uint32_t>(1, std::min<std::uint32_t>(numThreads, N));
  const auto countSlices = buildSlices(N, countingThreads);
  if (countSlices.empty())
    return;

  const std::size_t sliceCount = countSlices.size();
  if (totalBins >
      std::numeric_limits<std::size_t>::max() / std::max<std::size_t>(1, sliceCount)) {
    throw std::bad_alloc();
  }
  const std::size_t histogramSize = sliceCount * static_cast<std::size_t>(totalBins);
  std::vector<std::uint32_t> localCounts(histogramSize, 0);

  std::vector<std::uint32_t> binIndexTable(totalBins);
  for (std::uint64_t idx = 0; idx < totalBins; ++idx)
    binIndexTable[idx] = static_cast<std::uint32_t>(idx);

  auto countWorker = [&](std::size_t tid, const Slice &sl) {
    const row_t *rows = tbl.tuples + sl.begin;
    const std::size_t len = static_cast<std::size_t>(sl.end - sl.begin);
    std::uint32_t *local = localCounts.data() + tid * static_cast<std::size_t>(totalBins);
    constexpr std::size_t kBlockSize = 64;
    std::array<std::uint32_t, kBlockSize> blockBins{};

    for (std::size_t blockStart = 0; blockStart < len; blockStart += kBlockSize) {
      const std::size_t blockLen = std::min(kBlockSize, len - blockStart);
      for (std::size_t r = 0; r < blockLen; ++r) {
        const row_t &row = rows[blockStart + r];
        const std::uint32_t partId = row.hashKey & partitionMask;
        std::uint32_t binId;
        if (binsPowerOfTwo) {
          binId = (row.hashKey >> NUM_RADIX_BITS) & binMask;
        } else {
          binId = (row.hashKey >> NUM_RADIX_BITS) % binsPerPartition;
        }
        blockBins[r] = static_cast<std::uint32_t>(
            static_cast<std::uint64_t>(partId) * binsPerPartition + binId);
      }

#if defined(__AVX2__)
      const __m256i oneVec = _mm256_set1_epi32(1);
      std::uint64_t idx = 0;
      for (; idx + 8 <= totalBins; idx += 8) {
        const __m256i ids = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
            binIndexTable.data() + idx));
        __m256i acc = _mm256_setzero_si256();
        for (std::size_t r = 0; r < blockLen; ++r) {
          const __m256i target = _mm256_set1_epi32(static_cast<int>(blockBins[r]));
          const __m256i mask = _mm256_cmpeq_epi32(ids, target);
          acc = _mm256_add_epi32(acc, _mm256_and_si256(mask, oneVec));
        }
        __m256i counts =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(local + idx));
        counts = _mm256_add_epi32(counts, acc);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(local + idx), counts);
      }
      for (; idx < totalBins; ++idx) {
        std::uint32_t acc = 0;
        for (std::size_t r = 0; r < blockLen; ++r)
          acc += static_cast<std::uint32_t>(blockBins[r] == binIndexTable[idx]);
        local[idx] += acc;
      }
#else
      for (std::uint64_t idx = 0; idx < totalBins; ++idx) {
        std::uint32_t acc = 0;
        for (std::size_t r = 0; r < blockLen; ++r)
          acc += static_cast<std::uint32_t>(blockBins[r] == binIndexTable[idx]);
        local[idx] += acc;
      }
#endif
    }
  };

  parallel_slices(countSlices, [&](std::size_t tid, const Slice &sl) { countWorker(tid, sl); });

  std::vector<std::uint32_t> binCounts(totalBins, 0);
  const std::uint32_t binThreadCount =
      std::max<std::uint32_t>(1, std::min<std::uint32_t>(numThreads, (std::uint32_t)totalBins));
  const auto binSlices = buildSlices(static_cast<std::uint32_t>(totalBins), binThreadCount);
  parallel_slices(binSlices, [&](std::size_t tid, const Slice &sl) {
    for (std::uint32_t b = sl.begin; b < sl.end; ++b) {
      std::uint32_t sum = 0;
      std::size_t offset = static_cast<std::size_t>(b);
      for (std::size_t slice = 0; slice < sliceCount; ++slice) {
        sum += localCounts[offset];
        offset += static_cast<std::size_t>(totalBins);
      }
      binCounts[b] = sum;
    }
  });

  // ---- Compute per-bin deficits (secret) and prefix-sum starts (secret) ----
  std::vector<std::uint32_t> deficits(totalBins, 0);
  std::vector<std::uint32_t> starts(totalBins, 0);

  // Detect overflow (any bin > perBin). This should not happen if perBin is a
  // correct public upper bound; if it does, we fail fast.
  std::uint32_t anyOverflow = 0;

  parallel_slices(binSlices, [&](std::size_t /*tid*/, const Slice &sl) {
    for (std::uint32_t b = sl.begin; b < sl.end; ++b) {
      const std::uint32_t c = binCounts[b];
      const std::uint32_t overflow = bool_to_u32(c > perBin);
      // deficit = perBin - min(c, perBin)
      const std::uint32_t capped = ct_select_u32(c, perBin, overflow);
      const std::uint32_t def = perBin - capped;
      deficits[b] = def;
      // Accumulate a public-ish overflow flag (still depends on secret counts).
      // If it triggers, the algorithm cannot be correct anyway.
      __sync_fetch_and_or(&anyOverflow, overflow);
    }
  });

  if (anyOverflow) {
    throw std::runtime_error(
        "padTableUniformNested: overflow (some bin count > perBin); increase perBin or reduce bins");
  }

  // Parallel prefix sum over deficits (data-dependent values, fixed work).
  const std::uint32_t pfxThreads =
      std::max<std::uint32_t>(1, std::min<std::uint32_t>(numThreads, (std::uint32_t)totalBins));
  const auto pfxSlices = buildSlices(static_cast<std::uint32_t>(totalBins), pfxThreads);
  std::vector<std::uint32_t> sliceSums(pfxSlices.size(), 0);

  parallel_slices(pfxSlices, [&](std::size_t tid, const Slice &sl) {
    std::uint32_t run = 0;
    for (std::uint32_t b = sl.begin; b < sl.end; ++b)
      run += deficits[b];
    sliceSums[tid] = run;
  });

  std::vector<std::uint32_t> sliceOffsets(pfxSlices.size(), 0);
  {
    std::uint32_t run = 0;
    for (std::size_t t = 0; t < pfxSlices.size(); ++t) {
      sliceOffsets[t] = run;
      run += sliceSums[t];
    }
  }

  parallel_slices(pfxSlices, [&](std::size_t tid, const Slice &sl) {
    std::uint32_t run = sliceOffsets[tid];
    for (std::uint32_t b = sl.begin; b < sl.end; ++b) {
      starts[b] = run;
      run += deficits[b];
    }
  });

  // Optional consistency check (reveals only "something is wrong").
#ifndef NDEBUG
  {
    std::uint32_t sum = 0;
    for (std::uint32_t b = 0; b < (std::uint32_t)totalBins; ++b)
      sum += deficits[b];
    if (sum != totalDummies) {
      throw std::runtime_error("padTableUniformNested: deficit sum != totalDummies");
    }
  }
#endif

  // ---- Materialize dummy rows by ODistribute + forward fill ----
  // Work buffer must hold at least one seed candidate per bin, even when
  // totalDummies < totalBins.
  const std::uint32_t workLen =
      std::max<std::uint32_t>(totalDummies, static_cast<std::uint32_t>(totalBins));
  std::vector<row_t> work(static_cast<std::size_t>(workLen));

  // Initialize work to "empty" rows (cntExpand=0 means no value for ODistribute).
  {
    row_t empty{};
    // For rows with cntExpand==0, idx must be set so that obli_distribute_rows
    // does not treat them as "real" destinations. The oblivious distributor
    // expects dummies to use idx==UINT32_MAX (see prefix_sum_expand.h usage).
    empty.idx = std::numeric_limits<std::uint32_t>::max();
    empty.cntSelf = 0;
    empty.cntExpand = 0;
    empty.hashKey = 0;
    empty.shuffledIdx = 0;
    std::memset(empty.pay, 0, sizeof(empty.pay));

    const auto fillSlices = buildSlices(workLen, std::min(numThreads, workLen));
    parallel_slices(fillSlices, [&](std::size_t /*tid*/, const Slice &sl) {
      for (std::uint32_t i = sl.begin; i < sl.end; ++i)
        work[i] = empty;
    });
  }

  // Seed candidates: for each bin b, place one dummy with cntExpand = (deficit[b]!=0)
  // and destination idx = starts[b]. ODistribute places those seeds at their idx,
  // then a forward scan fills the gaps, producing exactly deficit[b] copies per bin.
  const auto seedSlices = buildSlices(static_cast<std::uint32_t>(totalBins),
                                      std::min<std::uint32_t>(numThreads,
                                                             static_cast<std::uint32_t>(totalBins)));
  parallel_slices(seedSlices, [&](std::size_t /*tid*/, const Slice &sl) {
    for (std::uint32_t b = sl.begin; b < sl.end; ++b) {
      const std::uint32_t partId = b / binsPerPartition;
      const std::uint32_t binId = b - partId * binsPerPartition;
      row_t d = make_dummy_row(partId, binId, binsPerPartition);

      const std::uint32_t hasDef = bool_to_u32(deficits[b] != 0u); // secret
      // For non-seed rows (hasDef==0), idx must be UINT32_MAX so they don't
      // affect obli_distribute_rows' internal counts. For seeds, idx is the
      // secret start position within [0, totalDummies).
      d.idx = ct_select_u32(std::numeric_limits<std::uint32_t>::max(), starts[b], hasDef);
      d.cntExpand = hasDef; // selection flag for ODistribute + carry-forward

      // Place seeds into the front part of work (independent of starts/deficits).
      work[b] = d;
    }
  });

  // Obliviously move seed rows to their destination indices.
  obli_distribute_rows(work.data(), workLen, numThreads);

  // Forward fill within the first totalDummies slots.
  table_t dummyTbl;
  dummyTbl.tuples = work.data();
  dummyTbl.num_tuples = totalDummies;
  carryForwardParallel(dummyTbl, buildSlices(totalDummies, std::min(numThreads, totalDummies)));

  // Canonicalize dummy fields for append.
  {
    const auto dSlices = buildSlices(totalDummies, std::min(numThreads, totalDummies));
    parallel_slices(dSlices, [&](std::size_t /*tid*/, const Slice &sl) {
      for (std::uint32_t i = sl.begin; i < sl.end; ++i) {
        work[i].cntSelf = 0;
        work[i].cntExpand = 0;
        work[i].idx = std::numeric_limits<std::uint32_t>::max();
        work[i].shuffledIdx = 0;
        std::memset(work[i].pay, 0, sizeof(work[i].pay));
      }
    });
  }

  // ---- Append dummies to original table ----
  row_t *expanded = new row_t[targetSize];

  // Copy the original tuples (public range [0..N)).
  {
    const auto realSlices = buildSlices(N, std::min(numThreads, N));
    parallel_slices(realSlices, [&](std::size_t /*tid*/, const Slice &sl) {
      const std::uint32_t len = sl.end - sl.begin;
      if (len == 0)
        return;
      std::memcpy(expanded + sl.begin, tbl.tuples + sl.begin,
                  static_cast<std::size_t>(len) * sizeof(row_t));
    });
  }

  // Append dummies (public range [N..targetSize)).
  {
    const auto dummySlices = buildSlices(totalDummies, std::min(numThreads, totalDummies));
    parallel_slices(dummySlices, [&](std::size_t /*tid*/, const Slice &sl) {
      const std::uint32_t len = sl.end - sl.begin;
      if (len == 0)
        return;
      std::memcpy(expanded + N + sl.begin, work.data() + sl.begin,
                  static_cast<std::size_t>(len) * sizeof(row_t));
    });
  }

  delete[] tbl.tuples;
  tbl.tuples = expanded;
  tbl.num_tuples = targetSize;
}

// Alias to make call sites explicit when both padding implementations are in
// scope.
inline void padTableUniformNested(table_t &tbl, std::uint32_t binsPerPartition,
                                  std::uint32_t numThreads) {
  padTableUniform(tbl, binsPerPartition, numThreads);
}
