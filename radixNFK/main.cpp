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

/*
 * Define this macro if your process is being killed due to insufficient memory.
 * Only required for 2^30 synthetic dataset if using the paper's configuration.
 * On low-RAM systems, you may need to run corresponding steps in R & S
 * sequentially (using numThreads) for large datasets.
 */
// #define INSUFFICIENT_MEMORY

// Global timer
std::chrono::high_resolution_clock::time_point tStart;

std::uint32_t L1_CACHE_SIZE = 1.1 * 1024 * 1024;
std::uint32_t BLOCK_SIZE = 32;

int main(int argc, char *argv[]) {
  printf("Set number of radix bits and passes for your workload in "
         "external/radix_partition/CMakeLists.txt.\n");
  std::uint32_t numThreads = 32;
  std::string inputPath = "../amazon.txt";

  if (argc > 1)
    numThreads = std::max<std::uint32_t>(1, std::stoul(argv[1]));
  if (argc > 2)
    inputPath = argv[2];
  if (argc > 3) {
    std::cerr << "Program takes 2 arguments: number of threads and input "
                 "filepath."
              << std::endl;
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

  std::uint32_t thrR = std::max<std::uint32_t>(
      1, ceil((static_cast<double>(t0.size()) / (t0.size() + t1.size())) *
              numThreads));
  std::uint32_t thrS = std::max<std::uint32_t>(1, numThreads - thrR);
  printf("thrR: %u, thrS: %u\n", thrR, thrS);

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

  auto slices_R = buildSlices(R.num_tuples, thrR);
  auto slices_S = buildSlices(S.num_tuples, thrS);

  std::uint32_t m;

  std::uint32_t num_radix_bits = static_cast<std::uint32_t>(
      std::log2((R.num_tuples * BLOCK_SIZE) / L1_CACHE_SIZE));
  std::uint32_t num_passes = (num_radix_bits > 5) ? 2 : 1;
  printf("(EXCHANGE)   radix bits: %2u, passes: %u\n", num_radix_bits,
         num_passes);

#define NUM_RADIX_BITS num_radix_bits
#define NUM_PASSES num_passes

#ifndef PRE_SORTED
  extern size_t total_num_threads;
  total_num_threads = numThreads;
  thread_system_init();

  std::vector<std::thread> pool;
  for (size_t i = 1; i < numThreads; ++i)
    pool.emplace_back(thread_start_work);

  tStart = std::chrono::high_resolution_clock::now();
  bitonic_sort_(R.tuples, true, 0, R.num_tuples, numThreads, false);
  bitonic_sort_(S.tuples, true, 0, S.num_tuples, numThreads, false);

  thread_release_all();
  for (auto &t : pool)
    t.join();
  thread_system_cleanup();
#else
  tStart = std::chrono::high_resolution_clock::now();
#endif

  std::thread partitionR([&] {
    std::vector<int> lastLen(slices_R.size()), mergeVal(slices_R.size() - 1);
    parallelCounts(R, slices_R, lastLen, mergeVal);
    replaceWithDummiesParallel(R, slices_R);
  });
  std::thread partitionS([&] {
    std::vector<int> lastLen(slices_S.size()), mergeVal(slices_S.size() - 1);
    parallelCounts(S, slices_S, lastLen, mergeVal);
    replaceWithDummiesParallel(S, slices_S);
  });
  partitionR.join();
  partitionS.join();

  RHO(&R, &S, numThreads);

  std::thread processR([&] {
    backfillDummiesParallel(R, slices_R);
    m = prefixSumExpandParallel(R, slices_R);
  });
  std::thread processS([&] {
    backfillDummiesParallel(S, slices_S);
    m = prefixSumExpandParallel(S, slices_S);
  });
  processR.join();
  processS.join();

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

#undef NUM_RADIX_BITS
#undef NUM_PASSES

  if (m < R.num_tuples) {
    num_radix_bits =
        static_cast<std::uint32_t>(std::log2((m * BLOCK_SIZE) / L1_CACHE_SIZE));
    num_passes = (num_radix_bits > 5) ? 2 : 1;
  }
  printf("(DISTRIBUTE) radix bits: %2u, passes: %u\n", num_radix_bits,
         num_passes);

#define NUM_RADIX_BITS num_radix_bits
#define NUM_PASSES num_passes

#ifndef INSUFFICIENT_MEMORY
  std::vector<Slice> slices_mR =
      buildSlices(m, std::max<std::uint32_t>(1, numThreads / 2));
  std::vector<Slice> slices_mS = buildSlices(
      m, std::max<std::uint32_t>(
             1, numThreads - (std::max<std::uint32_t>(1, numThreads / 2))));

  std::thread radixR([&] {
    if (m >= R.num_tuples) {
      RHO_idx(&R, &idxTable, thrR, &expandedR, true);
    } else {
      RHO_idx(&idxTable, &R, thrR, &expandedR, false);
    }
    carryForwardParallel(expandedR, slices_mR);
  });
  std::thread radixS([&] {
    if (m >= S.num_tuples) {
      RHO_idx(&S, &idxTable, thrS, &expandedS, true);
    } else {
      RHO_idx(&idxTable, &S, thrS, &expandedS, false);
    }
    carryForwardParallel(expandedS, slices_mS);
  });
  radixR.join();
  radixS.join();
#else
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
#endif

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
