#pragma once
#include <cstdint>
#include <thread>
#include <vector>

#include "slice_utils.h"
#include "inputs.h"

inline void assign_indices_parallel(table_t& tbl, std::uint32_t numThreads) {
  const size_t n = tbl.num_tuples;
  if (n == 0) return;
  auto slices = buildSlices(n, std::max<std::uint32_t>(1, numThreads));
  std::vector<std::thread> pool;
  pool.reserve(slices.size());
  for (const Slice& sl : slices) {
    pool.emplace_back([&, sl] {
      for (uint32_t i = sl.begin; i < sl.end; ++i) {
        tbl.tuples[i].idx = i;
      }
    });
  }
  for (auto& th : pool) th.join();
}
