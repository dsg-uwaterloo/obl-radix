#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <immintrin.h>
#include <inttypes.h>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <sys/random.h>
#include <sys/types.h>

#include "align_table.h"
#include "backfill_dummies.h"
#include "carry_forward.h"
#include "external_memory/algorithm/kway_butterfly_sort.hpp"
#include "inputs.h"
#include "merge.h"
#include "oblivious_ops.h"
#include "pad_table_sort.h"
#include "parallel_counts.h"
#include "parallel_index.h"
#include "prefix_sum_expand.h"
#include "raw_vector.h"
#include "replace_dummies.h"
#include "slice_utils.h"

extern "C" {
#include "bitonic.h"
#include "radix_join_counts.h"
#include "threading.h"
}

// Offline processing toggles:
// - Define `PROCESS_R` to run the offline pipeline for table R.
// - Define `PROCESS_S` to run the offline pipeline for table S.
// If both are defined, the program runs both pipelines (R then S) and reports
// a single combined "Offline" time.
#define PROCESS_R
#define PROCESS_S

// #define PRE_SORTED // use this if your tables are already sorted

#if !defined(PROCESS_R) && !defined(PROCESS_S)
#error "Define at least one of PROCESS_R or PROCESS_S"
#endif

static void shuffleTable(table_t &tbl, std::uint32_t numThreads) {
  if (tbl.num_tuples == 0)
    return;
  RawVector<row_t> vec{tbl.tuples, static_cast<std::size_t>(tbl.num_tuples)};
  try {
    if (vec.size() <= 512) {
      ::StdVector<row_t> tmp(vec.size());
      std::copy(vec.begin(), vec.end(), tmp.begin());
      EM::Algorithm::OrShuffle(tmp);
      std::copy(tmp.begin(), tmp.end(), vec.begin());
      return;
    }
    // Provide a heap size proportional to the data; back off on bad_alloc.
    constexpr uint64_t MAX_HEAP = 190ULL << 30;
    constexpr uint64_t MIN_HEAP = 64ULL << 20;
    uint64_t heapSize = std::max<uint64_t>(vec.size(), 4096UL) *
                        sizeof(EM::Algorithm::TaggedT<row_t>) * 2;
    heapSize = std::max<uint64_t>(heapSize, MIN_HEAP);
    heapSize = std::min<uint64_t>(heapSize, MAX_HEAP);

    while (true) {
      try {
        EM::Algorithm::KWayButterflyOShuffleFixedThreads(
            vec.begin(), vec.end(), 0, heapSize, (int)numThreads);
        break;
      } catch (const std::bad_alloc &) {
        if (heapSize <= MIN_HEAP) {
          throw;
        }
        heapSize /= 2;
      }
    }
  } catch (const std::exception &e) {
    fprintf(stderr,
            "Shuffle failed (%s) at size %zu; using StdVector OrShuffle "
            "fallback\n",
            e.what(), static_cast<size_t>(tbl.num_tuples));
    ::StdVector<row_t> tmp(vec.size());
    std::copy(vec.begin(), vec.end(), tmp.begin());
    EM::Algorithm::OrShuffle(tmp);
    std::copy(tmp.begin(), tmp.end(), vec.begin());
  }
}

inline __m128i generateSecret() {
  alignas(16) __m128i secret; // 128-bit aligned buffer
  ssize_t ret = getrandom(&secret, sizeof(secret), 0);
  if (ret != sizeof(secret))
    throw std::runtime_error("getrandom failed");
  return secret;
}

int main(int argc, char *argv[]) {
  printf("[INFO] Set BINS_PER_PART / NUM_RADIX_BITS / NUM_PASSES in the "
         "top-level CMakeLists.txt (or via -D... at CMake configure time).\n");
  constexpr std::uint32_t numThreads = 32;
  std::string inputPath = "../../datasets/real/amazon.txt";

  if (argc > 1)
    inputPath = argv[1];
  if (argc > 2) {
    std::cerr << "Program takes 1 argument: input filepath." << std::endl;
    return 1;
  }
  printf("Input: %s\n", inputPath.c_str());
  printf("Threads: %u\n", numThreads);

  __m128i SECRET = generateSecret();

  std::vector<Record> t0, t1;
  if (!load_two_tables(inputPath, t0, t1))
    exit(1);

  if (t0.size() > t1.size())
    std::swap(t0, t1);

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

  printf("\nRadix bits: %u, Passes: %u\n", NUM_RADIX_BITS, NUM_PASSES);
  const std::uint32_t bins = BINS_PER_PART;
  if ((bins == 0) || (bins & (bins - 1u)) != 0u) {
    std::cerr << "Bins must be a non-zero power-of-two." << std::endl;
    return 1;
  }
  printf("Bins: %u\n", bins);

#ifndef PRE_SORTED
  total_num_threads = numThreads;
  thread_system_init();

  std::vector<std::thread> pool;
  for (size_t i = 1; i < numThreads; ++i)
    pool.emplace_back(thread_start_work);

  std::chrono::high_resolution_clock::time_point tStart =
      std::chrono::high_resolution_clock::now();
#ifdef PROCESS_R
  bitonic_sort_(R.tuples, true, 0, R.num_tuples, numThreads, false);
#endif
#ifdef PROCESS_S
  bitonic_sort_(S.tuples, true, 0, S.num_tuples, numThreads, false);
#endif
  thread_release_all();
  for (auto &t : pool)
    t.join();
  thread_system_cleanup();
#else
  std::chrono::high_resolution_clock::time_point tStart =
      std::chrono::high_resolution_clock::now();
#endif

#ifdef PROCESS_R
  const auto slices_R = buildSlices(R.num_tuples, numThreads);
  {
    std::vector<int> lastLen(slices_R.size()), mergeVal(slices_R.size() - 1);
    parallelCounts(R, slices_R, lastLen, mergeVal);
    replaceWithDummiesParallel(R, slices_R, SECRET);
  }
#endif

#ifdef PROCESS_S
  const auto slices_S = buildSlices(S.num_tuples, numThreads);
  {
    std::vector<int> lastLen(slices_S.size()), mergeVal(slices_S.size() - 1);
    parallelCounts(S, slices_S, lastLen, mergeVal);
    replaceWithDummiesParallel(S, slices_S, SECRET);
  }
#endif

  {
    PadTableSortContext padCtx(bins, numThreads);
#ifdef PROCESS_R
    padCtx.pad(R);
#endif

#ifdef PROCESS_S
    padCtx.pad(S);
#endif
  }

#ifdef PROCESS_R
  shuffleTable(R, numThreads);
  assign_indices_parallel(R, numThreads);
#endif

#ifdef PROCESS_S
  shuffleTable(S, numThreads);
  assign_indices_parallel(S, numThreads);
#endif

  std::chrono::high_resolution_clock::time_point tEnd =
      std::chrono::high_resolution_clock::now();

  double sec =
      std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart)
          .count();
  printf("\nOffline: %f s\n", sec);

  delete[] R.tuples;
  delete[] S.tuples;
  return 0;
}
