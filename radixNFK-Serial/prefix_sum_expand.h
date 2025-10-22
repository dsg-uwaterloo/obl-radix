#pragma once
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "data-types.h"
#include "slice_utils.h"
#include "triple32.h"

inline uint64_t prefixSumExpandParallel(table_t &tbl,
                                        const std::vector<Slice> &slices) {
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

      for (uint32_t i = sl.begin; i < sl.end; ++i) {
        row_t &rec = tbl.tuples[i];
        uint32_t cnt = rec.cntExpand;
        uint32_t mask = -(cnt != 0);

        uint32_t dummyVal = triple32(i);
        uint32_t idx0 = (mask & run) | (~mask & dummyVal);

        rec.idx = idx0;
        rec.hashKey = dummyVal;
        run += cnt;
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
  const uint32_t m = running;

  pool.clear();
  pool.reserve(P);
  for (size_t t = 0; t < P; ++t) {
    pool.emplace_back([&, t] {
      const Slice sl = slices[t];
      uint32_t off32 = offset[t];

      for (size_t i = sl.begin; i < sl.end; ++i) {
        row_t &rec = tbl.tuples[i];
        uint32_t cnt = rec.cntExpand;
        uint32_t mask = -(cnt != 0);

        uint32_t newIdx = rec.idx + (mask & off32);
        rec.idx = newIdx;

        uint32_t newHash = triple32(newIdx);
        rec.hashKey = newHash;
      }
    });
  }
  for (auto &th : pool)
    th.join();

  return m;
}
