#pragma once
#include "inputs.h"
#include "slice_utils.h"
#include <cstring>
#include <thread>
#include <vector>

struct JoinRec {
  std::uint32_t keyR;
  std::uint32_t keyS;
  char payR[DATA_LENGTH];
  char payS[DATA_LENGTH];
} __attribute__((aligned(32)));

inline void mergeExpandedParallel(const table_t &expandedR,
                                  const table_t &expandedS, unsigned numThreads,
                                  std::vector<JoinRec> &out) {
  const std::size_t N = expandedR.num_tuples;
  if (N == 0) {
    out.clear();
    return;
  }

  out.resize(N);

  auto slices = buildSlices(N, numThreads);
  std::vector<std::thread> pool;
  pool.reserve(slices.size());

  for (const Slice &sl : slices)
    pool.emplace_back([&, sl] {
      for (std::size_t i = sl.begin; i < sl.end; ++i) {
        out[i].keyR = expandedR.tuples[i].key;
        out[i].keyS = expandedS.tuples[i].key;
        std::memcpy(&out[i].payR, &expandedR.tuples[i].pay, DATA_LENGTH);
        std::memcpy(&out[i].payS, &expandedS.tuples[i].pay, DATA_LENGTH);
      }
    });
  for (auto &th : pool)
    th.join();
}
