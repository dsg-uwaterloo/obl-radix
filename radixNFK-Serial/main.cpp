#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <sys/mman.h>

#include "align_table.h"
#include "backfill_dummies.h"
#include "carry_forward.h"
#include "inputs.h"
#include "merge.h"
#include "parallel_counts.h"
#include "prefix_sum_expand.h"
#include "replace_dummies.h"
#include "result_indices.h"
#include "slice_utils.h"

extern "C" {
#include "bitonic.h"
#include "radix_join_counts.h"
#include "radix_join_idx.h"
#include "threading.h"
}

// #define PRE_SORTED // use this if your tables are already sorted

// Global timer
std::chrono::high_resolution_clock::time_point tStart;

int main(int argc, char *argv[]) {
  printf("Set number of radix bits and passes for your workload in "
         "external/radix_partition/CMakeLists.txt.\n");
  std::uint32_t numThreads = 1;
  std::string inputPath = "../amazon.txt";

  if (argc > 1)
    inputPath = argv[1];
  if (argc > 2) {
    std::cerr << "Program takes 1 argument: input filepath." << std::endl;
    return 1;
  }
  printf("Input   : %s\n", inputPath.c_str());
  printf("Threads : %u\n", numThreads);

  std::vector<Record> t0, t1;
  if (!load_two_tables(inputPath, t0, t1))
    return 1;

  if (t0.size() > t1.size())
    std::swap(t0, t1);

  std::vector<Record> partR;
  partR.reserve(t0.size());
  std::vector<Record> partS;
  partS.reserve(t1.size());

  table_t R, S;
  R.tuples = new row_t[t0.size()];
  std::memcpy(R.tuples, t0.data(), t0.size() * sizeof(Record));
  R.num_tuples = static_cast<uint32_t>(t0.size());

  S.tuples = new row_t[t1.size()];
  std::memcpy(S.tuples, t1.data(), t1.size() * sizeof(Record));
  S.num_tuples = static_cast<uint32_t>(t1.size());

  t0.clear();
  t0.shrink_to_fit();
  t1.clear();
  t1.shrink_to_fit();

  auto slices_R = buildSlices(R.num_tuples, numThreads);
  auto slices_S = buildSlices(S.num_tuples, numThreads);

  std::uint32_t m;
#ifndef PRE_SORTED
  tStart = std::chrono::high_resolution_clock::now();
  bitonic_sort_(R.tuples, true, 0, R.num_tuples, numThreads, false);
  bitonic_sort_(S.tuples, true, 0, S.num_tuples, numThreads, false);
#else
  tStart = std::chrono::high_resolution_clock::now();
#endif

  std::vector<int> lastLen(slices_R.size()), mergeVal(slices_R.size() - 1);
  parallelCounts(R, slices_R, lastLen, mergeVal);
  replaceWithDummiesParallel(R, slices_R);
  std::vector<int> lastLenS(slices_S.size()), mergeValS(slices_S.size() - 1);
  parallelCounts(S, slices_S, lastLenS, mergeValS);
  replaceWithDummiesParallel(S, slices_S);

  RHO(&R, &S, numThreads);

  backfillDummiesParallel(R, slices_R);
  m = prefixSumExpandParallel(R, slices_R);
  backfillDummiesParallel(S, slices_S);
  m = prefixSumExpandParallel(S, slices_S);

  std::vector<Slice> slices_m = buildSlices(m, numThreads);

  const std::size_t bytes = m * sizeof(row_t);
  table_t idxTable{};
  idxTable.tuples = static_cast<row_t *>(aligned_alloc(32, bytes));
  idxTable.num_tuples = m;
  buildResultIndices(slices_m, idxTable);

  table_t expandedR{}, expandedS{};
  expandedR.num_tuples = m;
  expandedS.num_tuples = m;
  expandedR.tuples = static_cast<row_t *>(std::aligned_alloc(32, bytes));
  expandedS.tuples = static_cast<row_t *>(std::aligned_alloc(32, bytes));
  std::memset(expandedR.tuples, 0, bytes);
  std::memset(expandedS.tuples, 0, bytes);

  if (m >= R.num_tuples) {
    RHO_idx(&R, &idxTable, numThreads, &expandedR, true);
  } else {
    RHO_idx(&idxTable, &R, numThreads, &expandedR, false);
  }
  carryForwardParallel(expandedR, slices_m);

  if (m >= S.num_tuples) {
    RHO_idx(&S, &idxTable, numThreads, &expandedS, true);
  } else {
    RHO_idx(&idxTable, &S, numThreads, &expandedS, false);
  }
  carryForwardParallel(expandedS, slices_m);

  alignTableParallel(expandedS, slices_m, numThreads);

  std::vector<JoinRec> joinResults;
  mergeExpandedParallel(expandedR, expandedS, numThreads, joinResults);
  {
    std::ofstream outER("join.txt");
    for (const auto &j : joinResults)
      outER << j.keyR << ' ' << j.payR << ' ' << j.keyS << ' ' << j.payS
            << '\n';
  }

  printf("Join result rows : %ld (written to join.txt)\n",
         expandedR.num_tuples);

  return 0;
}
