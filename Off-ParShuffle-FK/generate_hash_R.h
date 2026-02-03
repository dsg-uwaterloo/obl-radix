#pragma once
#include <cstdint>
#include <immintrin.h>
#include <thread>
#include <vector>

#include "data-types.h"
#include "inputs.h"
#include "replace_dummies.h"
#include "slice_utils.h"

using std::vector;

inline void generateHashParallel(table_t &table,
                                 const vector<Slice> &slices,
                                 const __m128i &secret)

{
  constexpr uint64_t kHashSalt = 0xFFFFFFFFFFFFFFFFULL; /* constant salt */
  const uint32_t P = slices.size();
  vector<std::thread> pool;
  pool.reserve(P);

  for (uint32_t t = 0; t < P; ++t) {
    pool.emplace_back([&, t] {
      PRF_AES128_fast prf(secret);
      const Slice sl = slices[t];
      uint32_t i = sl.begin;
      const uint32_t end = sl.end;

      for (; i + 4 <= end; i += 4) {
        if (i + 32 < end) {
          _mm_prefetch(reinterpret_cast<const char *>(&table.tuples[i + 32]),
                       _MM_HINT_T0);
        }

        table.tuples[i].hashKey = prf(table.tuples[i].key, kHashSalt);
        table.tuples[i + 1].hashKey = prf(table.tuples[i + 1].key, kHashSalt);
        table.tuples[i + 2].hashKey = prf(table.tuples[i + 2].key, kHashSalt);
        table.tuples[i + 3].hashKey = prf(table.tuples[i + 3].key, kHashSalt);
      }

      for (; i < end; ++i) {
        table.tuples[i].hashKey = prf(table.tuples[i].key, kHashSalt);
      }
    });
  }
  for (auto &th : pool)
    th.join();
}
