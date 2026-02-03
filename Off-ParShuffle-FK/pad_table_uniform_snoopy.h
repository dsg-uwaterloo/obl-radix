#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "external/radix_partition/data-types.h"
#include "oblivious_ops.h"
#include "slice_utils.h"

extern "C" {
#include "external/bitonic/bitonic.h"
#include "external/bitonic/threading.h"
}

// Implemented in `main.cpp`. `padTableUniform` depends on it to compute the
// public fixed bin size.
inline int elemPerBucket(uint64_t totalSize, uint64_t bucketCount);

namespace pad_table_uniform_detail {

inline bool is_power_of_two(std::uint32_t x) { return x && ((x & (x - 1u)) == 0); }

inline std::uint32_t log2_pow2(std::uint32_t x) {
  return static_cast<std::uint32_t>(__builtin_ctz(x));
}

inline std::uint32_t mask_lowbits(std::uint32_t bits) {
  // if (bits >= 32u)
  //   return 0xffffffffu;
  // if (bits == 0u)
  //   return 0u;
  return (1u << bits) - 1u;
}

inline std::uint32_t bool_to_u32(bool x) { return static_cast<std::uint32_t>(x); }

inline std::uint32_t ct_select_u32(std::uint32_t a, std::uint32_t b,
                                   std::uint32_t sel01) {
  const std::uint32_t mask = 0u - sel01; // 0x00000000 or 0xffffffff
  return (a & ~mask) | (b & mask);
}

template <class Func>
inline void parallel_slices(const std::vector<Slice> &slices, const Func &fn) {
  if (slices.empty())
    return;
  std::vector<std::thread> pool;
  pool.reserve(slices.size());
  for (const Slice &sl : slices) {
    pool.emplace_back([&, sl] { fn(sl); });
  }
  for (auto &th : pool)
    th.join();
}

} // namespace pad_table_uniform_detail

// Pads each (partition,bin) group to a fixed size computed from public
// parameters. This is done by appending exactly S extra dummies per bin, sorting
// by destination bin, and compacting to keep exactly S rows per bin.
inline void padTableUniform(table_t &tbl, std::uint32_t bins,
                            std::uint32_t numThreads) {
  using namespace pad_table_uniform_detail;
  numThreads = std::max<std::uint32_t>(1, numThreads);

  if (tbl.num_tuples == 0)
    return;
  if (!is_power_of_two(bins))
    throw std::runtime_error("padTableUniform: bins must be a power of two");

  constexpr std::uint32_t rBits = NUM_RADIX_BITS;
  const std::uint32_t bBits = log2_pow2(bins);
  const std::uint32_t totalBits = rBits + bBits;
  const std::uint32_t lowMask = mask_lowbits(totalBits);

  const std::uint64_t partitions = 1ull << rBits;
  const std::uint64_t totalBins =
      partitions * static_cast<std::uint64_t>(bins); // 2^(r+b)

  const std::uint32_t N = static_cast<std::uint32_t>(tbl.num_tuples);
  const std::uint32_t S =
      static_cast<std::uint32_t>(std::max<int>(1, elemPerBucket(N, totalBins)));

  const std::uint64_t totalDummies = totalBins * static_cast<std::uint64_t>(S);
  const std::uint64_t newN64 = static_cast<std::uint64_t>(N) + totalDummies;
  if (newN64 > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error("padTableUniform: input too large for bitonic sort");
  const std::uint32_t newN = static_cast<std::uint32_t>(newN64);

  row_t *expanded = new row_t[newN];
  std::memcpy(expanded, tbl.tuples, static_cast<std::size_t>(N) * sizeof(row_t));

  // Fill extra dummies in parallel; each dummy encodes a (partition,bin) in the
  // lowest (r+b) bits of hashKey and is identified by idx==UINT32_MAX.
  {
    const std::uint32_t fillThreads =
        std::min<std::uint32_t>(numThreads, static_cast<std::uint32_t>(totalDummies));
    auto slices = buildSlices(static_cast<std::uint32_t>(totalDummies),
                              std::max<std::uint32_t>(1, fillThreads));
    parallel_slices(slices, [&](const Slice &sl) {
      for (std::uint64_t t = sl.begin; t < sl.end; ++t) {
        const std::uint64_t binIndex = t / static_cast<std::uint64_t>(S);
        const std::uint32_t part =
            static_cast<std::uint32_t>(binIndex / static_cast<std::uint64_t>(bins));
        const std::uint32_t bin =
            static_cast<std::uint32_t>(binIndex - static_cast<std::uint64_t>(part) * bins);
        row_t d{};
        d.idx = std::numeric_limits<std::uint32_t>::max();
        d.hashKey = (part | (bin << rBits)) & lowMask;
        expanded[static_cast<std::uint32_t>(N + t)] = d;
      }
    });
  }

  delete[] tbl.tuples;
  tbl.tuples = expanded;
  tbl.num_tuples = newN;

  // Bitonic sort uses global state; serialize the call and allow it to use its
  // internal worker system.
  {
    total_num_threads = numThreads;
    thread_system_init();
    std::vector<std::thread> pool;
    pool.reserve(numThreads - 1);
    for (std::uint32_t i = 1; i < numThreads; ++i)
      pool.emplace_back(thread_start_work);

    bitonic_sort_hashkey_lowbits_(reinterpret_cast<elem_t *>(tbl.tuples), true,
                                 0, static_cast<int>(newN),
                                 static_cast<int>(numThreads), totalBits);

    thread_release_all();
    for (auto &t : pool)
      t.join();
    thread_system_cleanup();
  }

  // Compute position-within-bin using a parallel block scan (segmented scan)
  // based on the sorted low bits of hashKey, then keep only the first S entries
  // per bin.
  std::vector<std::uint32_t> pos(newN);
  const std::uint32_t blocks = std::min<std::uint32_t>(numThreads, newN);
  auto slices = buildSlices(newN, blocks);

  std::vector<std::uint32_t> startKey(blocks), endKey(blocks), prefixLen(blocks),
      suffixLen(blocks), blockLen(blocks);

  // First pass: per-block local positions and run metadata.
  {
    std::vector<std::thread> pool;
    pool.reserve(blocks);
    for (std::uint32_t t = 0; t < blocks; ++t) {
      const Slice sl = slices[t];
      pool.emplace_back([&, t, sl] {
        const std::uint32_t begin = sl.begin;
        const std::uint32_t end = sl.end;
        const std::uint32_t len = end - begin;
        blockLen[t] = len;
        if (len == 0)
          return;

        auto keyAt = [&](std::uint32_t i) -> std::uint32_t {
          return tbl.tuples[i].hashKey & lowMask;
        };

        const std::uint32_t first = keyAt(begin);
        startKey[t] = first;
        std::uint32_t prev = first;
        std::uint32_t runPos = 1;
        pos[begin] = 1;

        std::uint32_t preLen = 1;
        std::uint32_t prefixDone = 0; // 0/1

        for (std::uint32_t i = begin + 1; i < end; ++i) {
          const std::uint32_t k = keyAt(i);
          const std::uint32_t samePrev = bool_to_u32(k == prev);
          const std::uint32_t sameMask = 0u - samePrev;
          runPos = (sameMask & (runPos + 1u)) | (~sameMask & 1u);
          pos[i] = runPos;

          const std::uint32_t sameFirst = bool_to_u32(k == first);
          const std::uint32_t stillPrefix = (1u - prefixDone) & sameFirst;
          preLen += stillPrefix;
          prefixDone |= (1u - sameFirst);

          prev = k;
        }

        endKey[t] = prev;
        prefixLen[t] = preLen;
        suffixLen[t] = runPos; // length of the last run in this block
      });
    }
    for (auto &th : pool)
      th.join();
  }

  // Second pass: compute carry offsets across blocks (public-size scan).
  std::vector<std::uint32_t> startOffset(blocks, 0), endRunTotal(blocks, 0);
  for (std::uint32_t t = 0; t < blocks; ++t) {
    const std::uint32_t hasPrev = bool_to_u32(t > 0);
    std::uint32_t prevNonEmpty = 1u;
    if (t > 0) {
      prevNonEmpty = bool_to_u32(blockLen[t - 1] != 0);
    }
    const std::uint32_t nonEmpty = bool_to_u32(blockLen[t] != 0) & prevNonEmpty;
    std::uint32_t keyMatch = 0u;
    if (t > 0) {
      keyMatch = bool_to_u32(startKey[t] == endKey[t - 1]);
    }
    const std::uint32_t carry = nonEmpty & keyMatch;
    std::uint32_t prevEnd = 0u;
    if (t > 0) {
      prevEnd = endRunTotal[t - 1];
    }
    startOffset[t] = carry * prevEnd;

    const std::uint32_t wholeBlockSame =
        bool_to_u32(blockLen[t] != 0) & bool_to_u32(prefixLen[t] == blockLen[t]) &
        bool_to_u32(startKey[t] == endKey[t]);
    const std::uint32_t extended = startOffset[t] + blockLen[t];
    endRunTotal[t] = ct_select_u32(suffixLen[t], extended, wholeBlockSame);
  }

  // Third pass: add carry only to the first run of each block.
  {
    std::vector<std::thread> pool;
    pool.reserve(blocks);
    for (std::uint32_t t = 0; t < blocks; ++t) {
      const Slice sl = slices[t];
      const std::uint32_t off = startOffset[t];
      const std::uint32_t pre = prefixLen[t];
      pool.emplace_back([&, sl, off, pre] {
        const std::uint32_t begin = sl.begin;
        const std::uint32_t end = sl.end;
        for (std::uint32_t i = begin; i < end; ++i) {
          const std::uint32_t inPrefix = bool_to_u32((i - begin) < pre);
          pos[i] += inPrefix * off;
        }
      });
    }
    for (auto &th : pool)
      th.join();
  }

  auto selected = std::make_unique<bool[]>(newN);
  {
    auto selSlices = buildSlices(newN, std::min<std::uint32_t>(numThreads, newN));
    parallel_slices(selSlices, [&](const Slice &sl) {
      for (std::uint32_t i = sl.begin; i < sl.end; ++i) {
        selected[i] = (pos[i] <= S);
      }
    });
  }

  obli_compact_rows(tbl.tuples, selected.get(), newN, numThreads);

  const std::uint64_t finalN64 = totalBins * static_cast<std::uint64_t>(S);
  const std::uint32_t finalN = static_cast<std::uint32_t>(finalN64);
  row_t *shrunk = new row_t[finalN];
  std::memcpy(shrunk, tbl.tuples,
              static_cast<std::size_t>(finalN) * sizeof(row_t));
  delete[] tbl.tuples;
  tbl.tuples = shrunk;
  tbl.num_tuples = finalN;
}
