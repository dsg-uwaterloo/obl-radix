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

// Global timers
std::chrono::high_resolution_clock::time_point tStart, tEnd;

// inspired from "bit twiddling hacks":
// http://graphics.stanford.edu/~seander/bithacks.html
inline uint32_t prevPow2(uint32_t v) {
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v - (v >> 1);
}

/**
 * Find the maximum number of bins that achieves a target probability
 * Lemma 1: m * exp(-n/m) ≈ target_p
 */
inline std::pair<std::uint32_t, double>
findMaxBins(double n, double target_p = 0.001, double eps = 1e-6) {
  int i;
  double low = 1, high = n, m = 0, p = 0;
  for (i = 0; i < 100; ++i) {
    m = (low + high) / 2.0;
    p = m * std::exp(-n / m);
    if (std::fabs(p - target_p) < eps)
      break;
    (p > target_p) ? (high = m) : (low = m);
  }

  if (i == 100) {
    std::cerr << "Lemma 1 unsatisfied. Reconfigure radix parameters."
              << std::endl;
  }

  return {prevPow2(static_cast<std::uint32_t>(std::ceil(m))), p};
}

int main(int argc, char *argv[]) {
  printf(
      "Set number of radix bits and passes in the top-level CMakeLists.txt.\n");
  std::uint32_t numThreads = 1;
  std::string inputPath = "../../datasets/real/amazon.txt";

  if (argc > 1)
    inputPath = argv[1];
  if (argc > 2) {
    std::cerr << "Program takes 1 argument: input filepath." << std::endl;
    return 1;
  }
  printf("Input: %s\n", inputPath.c_str());
  printf("Threads: %u\n", numThreads);

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

  printf("\nRadix bits: %u, Passes: %u\n", NUM_RADIX_BITS, NUM_PASSES);
  auto [bins, p] = findMaxBins(R.num_tuples / std::pow(2, NUM_RADIX_BITS));
  printf("(EXCHANGE)   Bins: %u, Lemma 1 p: %.4f\n", bins, p);

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

  RHO(&R, &S, numThreads, bins);

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
    RHO_idx(&R, &idxTable, numThreads, &expandedR, true, bins);
  } else {
    std::tie(bins, p) = findMaxBins(m / std::pow(2, NUM_RADIX_BITS));
    RHO_idx(&idxTable, &R, numThreads, &expandedR, false, bins);
  }
  carryForwardParallel(expandedR, slices_m);

  if (m >= S.num_tuples) {
    RHO_idx(&S, &idxTable, numThreads, &expandedS, true, bins);
  } else {
    RHO_idx(&idxTable, &S, numThreads, &expandedS, false, bins);
  }
  carryForwardParallel(expandedS, slices_m);

  alignTableParallel(expandedS, slices_m, numThreads);
  std::vector<JoinRec> joinResults;
  mergeExpandedParallel(expandedR, expandedS, numThreads, joinResults);

  printf("(DISTRIBUTE) Bins: %u, Lemma 1 p: %.4f\n", bins, p);
  double sec =
      std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart)
          .count();
  printf("\nJoin completed in %f s\n", sec);
  {
    std::ofstream outER("join.txt");
    for (const auto &j : joinResults)
      outER << j.keyR << ' ' << j.payR << ' ' << j.keyS << ' ' << j.payS
            << '\n';
  }
  printf("Join result rows: %ld (written to join.txt)\n", expandedR.num_tuples);

  return 0;
}
