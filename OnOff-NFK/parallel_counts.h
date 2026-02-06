#pragma once
#include "data-types.h"
#include "inputs.h"
#include "slice_utils.h"
#include <immintrin.h>
#include <thread>
#include <vector>

using std::vector;

inline void duplicateCountsParallel(table_t &data, const vector<Slice> &slices,
                                    vector<int> &lastLen) // |P|
{
  const std::size_t P = slices.size();
  if (P == 0)
    return;

  vector<std::thread> pool;
  pool.reserve(P);

  for (std::size_t t = 0; t < P; ++t) {
    pool.emplace_back([&, t] {
      const Slice sl = slices[t];
      std::uint32_t prev = data.tuples[sl.begin].key;
      int r = 1;
      std::size_t i = sl.begin + 1;
      const std::size_t end = sl.end;

      /* ---- 4-way unrolled run-length counter ---------------- */
      for (; i + 4 <= end; i += 4) {
        _mm_prefetch(reinterpret_cast<const char *>(&data.tuples[i + 16]),
                     _MM_HINT_T0);

        std::uint32_t curr0 = data.tuples[i].key;
        int match0 = -(curr0 == prev);
        data.tuples[i - 1].cntSelf = (~match0 & r);
        r = (match0 & (r + 1)) | (~match0 & 1);
        prev = curr0;

        std::uint32_t curr1 = data.tuples[i + 1].key;
        int match1 = -(curr1 == prev);
        data.tuples[i].cntSelf = (~match1 & r);
        r = (match1 & (r + 1)) | (~match1 & 1);
        prev = curr1;

        std::uint32_t curr2 = data.tuples[i + 2].key;
        int match2 = -(curr2 == prev);
        data.tuples[i + 1].cntSelf = (~match2 & r);
        r = (match2 & (r + 1)) | (~match2 & 1);
        prev = curr2;

        std::uint32_t curr3 = data.tuples[i + 3].key;
        int match3 = -(curr3 == prev);
        data.tuples[i + 2].cntSelf = (~match3 & r);
        r = (match3 & (r + 1)) | (~match3 & 1);
        prev = curr3;
      }

      /* ---- scalar tail (≤3 rows) --------------------------- */
      for (; i < end; ++i) {
        std::uint32_t curr = data.tuples[i].key;
        int match = -(curr == prev);
        data.tuples[i - 1].cntSelf = (~match & r);
        r = (match & (r + 1)) | (~match & 1);
        prev = curr;
      }
      data.tuples[end - 1].cntSelf = r;
      lastLen[t] = r;
    });
  }
  for (auto &th : pool)
    th.join();
}

/* -------------------------------------------------------------- */
inline void initializeMergeValsParallel(table_t &data,
                                        const vector<Slice> &slices,
                                        const vector<int> &lastLen,
                                        vector<int> &mergeVal) {
  const std::size_t P = slices.size();
  if (P < 2)
    return;

  vector<std::thread> pool;
  pool.reserve(P - 1);
  for (std::size_t t = 1; t < P; ++t) {
    pool.emplace_back([&, t] {
      std::uint32_t first = data.tuples[slices[t].begin].key;
      std::uint32_t last = data.tuples[slices[t - 1].end - 1].key;
      int match = -(first == last);
      mergeVal[t - 1] = match & lastLen[t - 1];
      std::size_t idx = slices[t - 1].end - 1;
      data.tuples[idx].cntSelf = (~match & data.tuples[idx].cntSelf);
    });
  }
  for (auto &th : pool)
    th.join();
}

inline void mergeDuplicatesParallel(const table_t &data,
                                    const vector<Slice> &slices,
                                    const vector<int> &lastLen,
                                    vector<int> &mergeVal) {
  const std::size_t P = slices.size();
  if (P < 3)
    return;

  vector<std::thread> pool;
  pool.reserve(P - 2);

  for (std::size_t i = 2; i < P; ++i) {
    pool.emplace_back([&, i] {
      std::uint32_t first = data.tuples[slices[i].begin].key;
      int acc = 0;
      for (std::size_t j = 0; j < i - 1; ++j) {
        std::uint32_t last = data.tuples[slices[j].end - 1].key;
        int match = -(first == last);
        acc += match & lastLen[j];
      }
      mergeVal[i - 1] += acc;
    });
  }
  for (auto &th : pool)
    th.join();
}

inline void finalizeCountsParallel(table_t &data, const vector<Slice> &slices,
                                   const vector<int> &mergeVal) {
  const std::size_t P = slices.size();
  if (P < 2)
    return;

  vector<std::thread> pool;
  pool.reserve(P - 1);
  for (std::size_t i = 1; i < P; ++i) {
    pool.emplace_back([&, i] {
      const Slice sl = slices[i];
      int done = 0;
      int mv = mergeVal[i - 1];
      for (std::size_t j = sl.begin; j < sl.end; ++j) {
        int isNZ = -(data.tuples[j].cntSelf != 0);
        int doAdd = (~done) & isNZ;
        data.tuples[j].cntSelf += doAdd & mv;
        done |= doAdd;
      }
    });
  }
  for (auto &th : pool)
    th.join();
}

inline void parallelCounts(table_t &data, const vector<Slice> &slices,
                           vector<int> &lastLen, vector<int> &mergeVal) {
  duplicateCountsParallel(data, slices, lastLen);
  initializeMergeValsParallel(data, slices, lastLen, mergeVal);
  mergeDuplicatesParallel(data, slices, lastLen, mergeVal);
  finalizeCountsParallel(data, slices, mergeVal);
}
