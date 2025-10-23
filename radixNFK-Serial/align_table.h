#pragma once
#include "inputs.h"
#include "slice_utils.h"
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

// Global timers defined in main.cpp
extern std::chrono::high_resolution_clock::time_point tStart, tEnd;

extern "C" {
#include "bitonic.h"
#include "threading.h"
}

inline void within_key(const table_t &tbl, const std::vector<Slice> &slices,
                       std::vector<int> &W) {
  const int N = static_cast<int>(tbl.num_tuples);
  W.resize(N);
  const int P = static_cast<int>(slices.size());

  std::vector<std::thread> pool;
  pool.reserve(P);

  std::vector<std::uint64_t> sliceFirstKey(P);
  std::vector<int> sliceOffsets(P);

  for (int t = 0; t < P; ++t)
    pool.emplace_back([&, t] {
      const Slice sl = slices[t];
      std::uint64_t prev = tbl.tuples[sl.begin].key;
      sliceFirstKey[t] = prev;
      int idx = 0;
      W[sl.begin] = idx;

      for (int i = sl.begin + 1; i < sl.end; ++i) {
        std::uint64_t cur = tbl.tuples[i].key;
        int same = -(cur == prev);
        int diff = ~same;
        idx = (same & (idx + 1)) | (diff & 0);
        prev = (same & prev) | (diff & cur);
        W[i] = idx;
      }
    });
  for (auto &th : pool)
    th.join();
  pool.clear();

  sliceOffsets[0] = 0;
  for (int t = 1; t < P; ++t) {
    int prevEnd = static_cast<int>(slices[t - 1].end) - 1;
    int curBeg = static_cast<int>(slices[t].begin);

    int spill = -(tbl.tuples[prevEnd].key == tbl.tuples[curBeg].key);
    int sameBeg = -(tbl.tuples[curBeg].key == sliceFirstKey[t - 1]);

    int tailLen = W[prevEnd] + 1;
    int offPrev = sameBeg & sliceOffsets[t - 1];
    sliceOffsets[t] = spill & (offPrev + tailLen);
  }

  for (int t = 0; t < P; ++t)
    pool.emplace_back([&, t] {
      const Slice sl = slices[t];
      int off = sliceOffsets[t];
      auto firstKey = sliceFirstKey[t];

      for (int i = sl.begin; i < sl.end; ++i) {
        int mask = -(tbl.tuples[i].key == firstKey);
        W[i] += mask & off;
      }
    });
  for (auto &th : pool)
    th.join();
}

inline void alignTableParallel(table_t &S, const std::vector<Slice> &slices,
                               unsigned numThreads) {
  const int N = static_cast<int>(S.num_tuples);
  if (N == 0)
    return;

  std::vector<int> W;
  within_key(S, slices, W);

  {
    std::vector<std::thread> pool;
    pool.reserve(slices.size());
    for (const Slice &sl : slices)
      pool.emplace_back([&, sl] {
        for (int i = sl.begin; i < sl.end; ++i) {
          int q = W[i];
          int a2 = S.tuples[i].cntExpand;
          int a1 = S.tuples[i].cntSelf;
          int row = q / a2;
          int col = q - row * a2;
          int ii = row + col * a1;
          S.tuples[i].idx = ii;
        }
      });
    for (auto &th : pool)
      th.join();
  }

  struct KeyIdxLess {
    bool operator()(const Record &a, const Record &b) const {
      return (a.key < b.key) || (a.key == b.key && a.idx < b.idx);
    }
  };

  thread_system_init();
  std::vector<std::thread> pool;
  for (size_t i = 1; i < numThreads; ++i)
    pool.emplace_back(thread_start_work);

  bitonic_sort_(reinterpret_cast<elem_t *>(S.tuples), true, 0, N, numThreads,
                true);

  tEnd = std::chrono::high_resolution_clock::now();
  thread_release_all();
  for (auto &t : pool)
    t.join();
  thread_system_cleanup();
}
