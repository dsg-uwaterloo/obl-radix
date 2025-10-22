#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <thread>
#include <vector>

#include "data-types.h"
#include "slice_utils.h"
#include "triple32.h"

using std::vector;

inline void buildResultIndices(const vector<Slice> &slices, table_t &idxOut) {
  vector<std::thread> pool;
  pool.reserve(slices.size());

  for (const Slice &sl : slices) {
    pool.emplace_back([&, sl] {
      for (std::uint32_t i = sl.begin; i < sl.end; ++i) {
        row_t &rec = idxOut.tuples[i];
        rec.idx = i;
        rec.hashKey = triple32(i);
      }
    });
  }
  for (auto &th : pool)
    th.join();
}
