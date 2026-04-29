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
#include <tbb/global_control.h>
#include <tbb/parallel_sort.h>

#include "backfill_dummies.h"
#include "external_memory/algorithm/kway_butterfly_sort.hpp"
#include "generate_hash_R.h"
#include "inputs.h"
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

// #define PRE_SORTED // use this if your tables are already sorted

// Global timer
std::chrono::high_resolution_clock::time_point tStart;

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
  printf("[INFO] R: Primary Key table; S: Foreign Key table\n");
  std::uint32_t numThreads = 32;
  std::string inputPath = "../../datasets/real/imdb/imdb.txt";

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
  printf("Input: %s\n", inputPath.c_str());
  printf("Threads: %u\n", numThreads);

  __m128i SECRET = generateSecret();

  std::vector<Record> t0, t1;
  if (!load_two_tables(inputPath, t0, t1))
    exit(1);

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
  const uint32_t originalSSize = S.num_tuples;

  t0.clear();
  t0.shrink_to_fit();
  t1.clear();
  t1.shrink_to_fit();

  auto slices_R_numThreads = buildSlices(R.num_tuples, numThreads);
  auto slices_S_numThreads = buildSlices(S.num_tuples, numThreads);

  std::vector<int> lastLen(slices_S_numThreads.size()),
      mergeVal(slices_S_numThreads.size() - 1);

  tbb::global_control c(tbb::global_control::max_allowed_parallelism,
                        numThreads);

  auto padTableToSize = [&](table_t &tbl, uint32_t target) {
    if (tbl.num_tuples == target)
      return;

    row_t *expanded = new row_t[target];
    const uint32_t copyCount = std::min<uint32_t>(tbl.num_tuples, target);
    std::memcpy(expanded, tbl.tuples, copyCount * sizeof(row_t));

    if (copyCount < target) {
      row_t dummy{};
      dummy.idx = UINT32_MAX;
      dummy.cntSelf = 0;
      const uint32_t fillCount = target - copyCount;
      const uint32_t fillThreads =
          std::min<std::uint32_t>(numThreads, fillCount);
      if (fillThreads <= 1) {
        std::fill(expanded + copyCount, expanded + target, dummy);
      } else {
        auto fillSlices = buildSlices(fillCount, fillThreads);
        std::vector<std::thread> pool;
        pool.reserve(fillSlices.size());
        for (const Slice &sl : fillSlices) {
          pool.emplace_back([&, sl] {
            std::fill(expanded + copyCount + sl.begin,
                      expanded + copyCount + sl.end, dummy);
          });
        }
        for (auto &th : pool) {
          th.join();
        }
      }
    }

    delete[] tbl.tuples;
    tbl.tuples = expanded;
    tbl.num_tuples = target;
  };

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

  tStart = std::chrono::high_resolution_clock::now();
  bitonic_sort_(S.tuples, true, 0, S.num_tuples, numThreads, false);

  thread_release_all();
  for (auto &t : pool)
    t.join();
  thread_system_cleanup();

  std::chrono::high_resolution_clock::time_point sortEnd =
      std::chrono::high_resolution_clock::now();
#else
  tStart = std::chrono::high_resolution_clock::now();
#endif

  parallelCounts(S, slices_S_numThreads, lastLen, mergeVal);
  replaceWithDummiesParallel(S, slices_S_numThreads, SECRET);
  generateHashParallel(R, slices_R_numThreads, SECRET); // UPDATE

  std::chrono::high_resolution_clock::time_point padStart =
      std::chrono::high_resolution_clock::now();
  {
    PadTableSortContext padCtx(bins, numThreads);
    padCtx.pad(R);
    padCtx.pad(S);
  }

  // printf("Padded R size = %u\n", R.num_tuples);
  // printf("Padded S size = %u\n", S.num_tuples);

  std::chrono::high_resolution_clock::time_point tShuffleStart =
      std::chrono::high_resolution_clock::now();
  shuffleTable(R, numThreads);
  shuffleTable(S, numThreads);
  assign_indices_parallel(S, numThreads);
  std::chrono::high_resolution_clock::time_point tShuffleEnd =
      std::chrono::high_resolution_clock::now();

  std::chrono::high_resolution_clock::time_point onStart =
      std::chrono::high_resolution_clock::now();
  if (S.num_tuples >= R.num_tuples) {
    RHO(&R, &S, numThreads, false, bins);
  } else {
    RHO(&S, &R, numThreads, true, bins);
  }
  std::chrono::high_resolution_clock::time_point exchangeEnd =
      std::chrono::high_resolution_clock::now();

  auto cmp = [](const row_t &a, const row_t &b) { return a.idx < b.idx; };
  tbb::parallel_sort(S.tuples, S.tuples + S.num_tuples, cmp);
  std::chrono::high_resolution_clock::time_point nonOblSortEnd =
      std::chrono::high_resolution_clock::now();

  S.num_tuples = originalSSize;

  backfillDummiesParallel(S, slices_S_numThreads);
  std::chrono::high_resolution_clock::time_point distEnd =
      std::chrono::high_resolution_clock::now();

  double onSec = std::chrono::duration_cast<std::chrono::duration<double>>(
                     distEnd - onStart)
                     .count();
  printf("\nOnline: %f s\n", onSec);

#ifndef ONLINE_ONLY
  double exchangeSec =
      std::chrono::duration_cast<std::chrono::duration<double>>(exchangeEnd -
                                                                onStart)
          .count();
  printf("\nPartitioning and exchanging counts took %f s\n", exchangeSec);

  double nonOblSortSec =
      std::chrono::duration_cast<std::chrono::duration<double>>(nonOblSortEnd -
                                                                exchangeEnd)
          .count();
  printf("\nNon-oblivious sort took %f s\n", nonOblSortSec);

  double distSec = std::chrono::duration_cast<std::chrono::duration<double>>(
                       distEnd - nonOblSortEnd)
                       .count();
  printf(
      "\nBackfill dummies, Prefix calculation, and Dist & Expand took %f s\n",
      distSec);

  double sec = std::chrono::duration_cast<std::chrono::duration<double>>(
                   distEnd - tStart)
                   .count();
  printf("\nOffline: %f s\n", sec - onSec);

#ifndef PRE_SORTED
  double sortSec = std::chrono::duration_cast<std::chrono::duration<double>>(
                       sortEnd - tStart)
                       .count();
  printf("\nSorting took %f s\n", sortSec);
#endif

  double paddingSec = std::chrono::duration_cast<std::chrono::duration<double>>(
                          tShuffleStart - padStart)
                          .count();
  printf("\nPadding took %f s\n", paddingSec);

  double shuffleSec = std::chrono::duration_cast<std::chrono::duration<double>>(
                          tShuffleEnd - tShuffleStart)
                          .count();
  printf("\nOShuffle took %f s\n", shuffleSec);
#endif

  {
    std::ofstream outER("join.txt");
    for (int i = 0; i < S.num_tuples; i++) {
      outER << S.tuples[i].key << ' ' << S.tuples[i].payPrimary << ' '
            << S.tuples[i].key << ' ' << S.tuples[i].paySelf << '\n';
    }
  }
  printf("\nJoin result rows: %ld (written to join.txt)\n", S.num_tuples);

  return 0;
}
