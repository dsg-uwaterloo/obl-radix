#pragma once
#include "data-types.h"
#include "inputs.h"
#include "slice_utils.h"
#include "triple32.h"
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <thread>
#include <vector>

using std::vector;

inline uint32_t generateDummy(uint32_t key, uint32_t index) {
  return triple32(key ^ index);
}

inline void replaceWithDummiesParallel(table_t &table,
                                       const vector<Slice> &slices)

{
  const uint32_t P = slices.size();
  vector<std::thread> pool;
  pool.reserve(P);

  for (uint32_t t = 0; t < P; ++t) {
    pool.emplace_back([&, t] {
      const Slice sl = slices[t];
      uint32_t i = sl.begin;
      const uint32_t end = sl.end;

      for (; i + 4 <= end; i += 4) {
        if (i + 32 < end) {
          _mm_prefetch(reinterpret_cast<const char *>(&table.tuples[i + 32]),
                       _MM_HINT_T0);
        }

        uint32_t d0 = generateDummy(table.tuples[i].key, i);
        uint32_t d1 = generateDummy(table.tuples[i + 1].key, i + 1);
        uint32_t d2 = generateDummy(table.tuples[i + 2].key, i + 2);
        uint32_t d3 = generateDummy(table.tuples[i + 3].key, i + 3);

        uint32_t m0 = -(table.tuples[i].cntSelf == 0);
        uint32_t m1 = -(table.tuples[i + 1].cntSelf == 0);
        uint32_t m2 = -(table.tuples[i + 2].cntSelf == 0);
        uint32_t m3 = -(table.tuples[i + 3].cntSelf == 0);

        table.tuples[i].key = (table.tuples[i].key & ~m0) | (d0 & m0);
        table.tuples[i + 1].key = (table.tuples[i + 1].key & ~m1) | (d1 & m1);
        table.tuples[i + 2].key = (table.tuples[i + 2].key & ~m2) | (d2 & m2);
        table.tuples[i + 3].key = (table.tuples[i + 3].key & ~m3) | (d3 & m3);

        table.tuples[i].idx = i;
        table.tuples[i + 1].idx = i + 1;
        table.tuples[i + 2].idx = i + 2;
        table.tuples[i + 3].idx = i + 3;

        table.tuples[i].hashKey = triple32(table.tuples[i].key);
        table.tuples[i + 1].hashKey = triple32(table.tuples[i + 1].key);
        table.tuples[i + 2].hashKey = triple32(table.tuples[i + 2].key);
        table.tuples[i + 3].hashKey = triple32(table.tuples[i + 3].key);
      }

      for (; i < end; ++i) {
        table.tuples[i].idx = i;
        uint32_t dummy = generateDummy(table.tuples[i].key, i);
        uint32_t mask = -(table.tuples[i].cntSelf == 0);
        table.tuples[i].key = (table.tuples[i].key & ~mask) | (dummy & mask);
        table.tuples[i].hashKey = triple32(table.tuples[i].key);
      }
    });
  }
  for (auto &th : pool)
    th.join();
}
