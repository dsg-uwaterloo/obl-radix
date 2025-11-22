#pragma once
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <openssl/aes.h>
#include <thread>
#include <vector>

#include "data-types.h"
#include "inputs.h"
#include "slice_utils.h"
#include "triple32.h"

using std::vector;

inline void generateHashParallel(table_t &table,
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

        table.tuples[i].hashKey = triple32(table.tuples[i].key);
        table.tuples[i + 1].hashKey = triple32(table.tuples[i + 1].key);
        table.tuples[i + 2].hashKey = triple32(table.tuples[i + 2].key);
        table.tuples[i + 3].hashKey = triple32(table.tuples[i + 3].key);
      }

      for (; i < end; ++i) {
        table.tuples[i].hashKey = triple32(table.tuples[i].key);
      }
    });
  }
  for (auto &th : pool)
    th.join();
}
