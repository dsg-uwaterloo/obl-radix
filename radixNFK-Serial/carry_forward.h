#pragma once
#include "inputs.h"
#include "slice_utils.h"
#include <algorithm>
#include <immintrin.h>
#include <thread>
#include <vector>

using std::vector;

inline void carryForwardParallel(table_t &tbl,
                                 const std::vector<Slice> &slices) {
  const std::size_t N = tbl.num_tuples;
  if (N == 0)
    return;
  const int P = slices.size();

  static thread_local std::vector<Record> lastScratch;
  static thread_local std::vector<Record> seedScratch;

  if (lastScratch.size() < static_cast<size_t>(P))
    lastScratch.resize(P);
  if (seedScratch.size() < static_cast<size_t>(P))
    seedScratch.resize(P);

  auto &last = lastScratch;
  auto &seed = seedScratch;

  struct row_t *tblTuples = tbl.tuples;
  std::vector<std::thread> pool;
  pool.reserve(P);
  for (std::size_t t = 0; t < P; ++t) {
    pool.emplace_back([=, &tblTuples, &last]() {
      const Slice &sl = slices[t];
      Record cur{};
      for (std::size_t i = sl.begin; i < sl.end; ++i) {
        if (i + 16 < sl.end)
          _mm_prefetch(reinterpret_cast<const char *>(&tblTuples[i + 16]),
                       _MM_HINT_T0);
        std::uint64_t isReal = -(tblTuples[i].cntSelf != 0);
        maskedCopyRecord32(reinterpret_cast<const Record *>(&tblTuples[i]),
                           &cur, isReal);
      }
      last[t] = cur;
    });
  }
  for (auto &th : pool)
    th.join();

  Record running{};
  for (int t = 0; t < P; ++t) {
    seed[t] = running;
    std::uint64_t has = -(last[t].cntSelf != 0);
    maskedCopyRecord32(reinterpret_cast<const Record *>(&last[t]), &running,
                       has);
  }

  pool.clear();
  pool.reserve(P);
  for (std::size_t t = 0; t < P; ++t) {
    pool.emplace_back([=, &tblTuples, &seed]() {
      const Slice &sl = slices[t];
      Record cur = seed[t];
      for (std::size_t i = sl.begin; i < sl.end; ++i) {
        if (i + 16 < sl.end)
          _mm_prefetch(reinterpret_cast<const char *>(&tblTuples[i + 16]),
                       _MM_HINT_T0);
        std::uint64_t isReal = -(tblTuples[i].cntSelf != 0);
        maskedCopyRecord32(reinterpret_cast<const Record *>(&tblTuples[i]),
                           &cur, isReal);
        maskedCopyRecord32(&cur, reinterpret_cast<Record *>(&tblTuples[i]),
                           std::numeric_limits<std::uint64_t>::max());
      }
    });
  }
  for (auto &th : pool)
    th.join();
}
