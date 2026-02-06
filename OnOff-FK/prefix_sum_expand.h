#pragma once
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "data-types.h"
#include "slice_utils.h"
// #include "triple32.h"

enum class PrefixSumExpandMode : std::uint8_t {
  CountsCntSelf = 0,     // run += cntSelf
  CountsShuffledIdx = 1, // run += shuffledIdx
  MatchFlag = 2          // run += 1 if payPrimary[0] != 0
};

inline uint64_t prefixSumExpandParallel(table_t &tbl,
                                        const std::vector<Slice> &slices,
                                        bool *selected = nullptr,
                                        PrefixSumExpandMode mode =
                                            PrefixSumExpandMode::CountsCntSelf) {
  const size_t N = tbl.num_tuples;
  if (N == 0)
    return 0;

  const size_t P = slices.size();

  std::vector<uint32_t> sliceSum(P, 0);
  std::vector<std::thread> pool;
  pool.reserve(P);

  for (size_t t = 0; t < P; ++t) {
    pool.emplace_back([&, t] {
      const Slice sl = slices[t];
      uint32_t run = 0;

      if (mode == PrefixSumExpandMode::CountsCntSelf) {
        for (uint32_t i = sl.begin; i < sl.end; ++i) {
          row_t &rec = tbl.tuples[i];
          const uint32_t cnt = rec.cntSelf;
          const bool keep = (cnt != 0);
          if (selected)
            selected[i] = keep;
          const uint32_t mask = -static_cast<uint32_t>(keep);

          const uint32_t dummyVal = UINT32_MAX;
          rec.idx = (mask & run) | (~mask & dummyVal);
          run += cnt;
        }
      } else if (mode == PrefixSumExpandMode::CountsShuffledIdx) {
        for (uint32_t i = sl.begin; i < sl.end; ++i) {
          row_t &rec = tbl.tuples[i];
          const uint32_t cnt = rec.shuffledIdx;
          const bool keep = (cnt != 0);
          if (selected)
            selected[i] = keep;
          const uint32_t mask = -static_cast<uint32_t>(keep);

          const uint32_t dummyVal = UINT32_MAX;
          rec.idx = (mask & run) | (~mask & dummyVal);
          run += cnt;
        }
      } else {
        for (uint32_t i = sl.begin; i < sl.end; ++i) {
          row_t &rec = tbl.tuples[i];
          const bool keep = rec.payPrimary[0] != 0;
          if (selected)
            selected[i] = keep;
          const uint32_t mask = -static_cast<uint32_t>(keep);

          const uint32_t dummyVal = UINT32_MAX;
          rec.idx = (mask & run) | (~mask & dummyVal);
          run += (-mask);
        }
      }
      sliceSum[t] = run;
    });
  }
  for (auto &th : pool)
    th.join();

  std::vector<uint32_t> offset(P);
  uint32_t running = 0;
  for (size_t t = 0; t < P; ++t) {
    offset[t] = running;
    running += sliceSum[t];
  }

  pool.clear();
  pool.reserve(P);
  for (size_t t = 0; t < P; ++t) {
    pool.emplace_back([&, t] {
      const Slice sl = slices[t];
      uint32_t off32 = offset[t];

      if (mode == PrefixSumExpandMode::CountsCntSelf) {
        for (size_t i = sl.begin; i < sl.end; ++i) {
          row_t &rec = tbl.tuples[i];
          const uint32_t cnt = rec.cntSelf;
          const uint32_t mask = -(cnt != 0);
          rec.idx = rec.idx + (mask & off32);
        }
      } else if (mode == PrefixSumExpandMode::CountsShuffledIdx) {
        for (size_t i = sl.begin; i < sl.end; ++i) {
          row_t &rec = tbl.tuples[i];
          const uint32_t cnt = rec.shuffledIdx;
          const uint32_t mask = -(cnt != 0);
          rec.idx = rec.idx + (mask & off32);
        }
      } else {
        for (size_t i = sl.begin; i < sl.end; ++i) {
          row_t &rec = tbl.tuples[i];
          const uint32_t mask = -(rec.payPrimary[0] != 0);
          rec.idx = rec.idx + (mask & off32);
        }
      }
    });
  }
  for (auto &th : pool)
    th.join();

  return running;
}
