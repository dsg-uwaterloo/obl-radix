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

// #define PRE_SORTED // use this if your tables are already sorted

// Global timer
std::chrono::high_resolution_clock::time_point tStart, tEnd;

static void printOutput(const char *label, const table_t &tbl) {
  printf("%s (num_tuples=%" PRIu64 ")\n", label,
         static_cast<uint64_t>(tbl.num_tuples));
  for (uint64_t i = 0; i < tbl.num_tuples; ++i) {
    const row_t &rec = tbl.tuples[i];
    printf("  [%" PRIu64 "] key=%u cntSelf=%u cntExpand=%u idx=%u hashKey=%u\n",
           i, rec.key, rec.cntSelf, rec.cntExpand, rec.idx, rec.hashKey);
  }
}

inline std::uint32_t prev_power_of_two(std::uint32_t n) {
  if (n <= 1)
    return 1;
  const unsigned leading = static_cast<unsigned>(__builtin_clz(n));
  return 1u << (31u - leading);
}

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
  std::uint32_t numThreads = 32;
  std::string inputPath = "../../datasets/real/amazon.txt";

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
  printf("threads_R: %u, threads_S: %u\n", thrR, thrS);

  table_t R, S;
  R.tuples = new row_t[t0.size()];
  std::memcpy(R.tuples, t0.data(), t0.size() * sizeof(Record));
  R.num_tuples = static_cast<uint32_t>(t0.size());

  S.tuples = new row_t[t1.size()];
  std::memcpy(S.tuples, t1.data(), t1.size() * sizeof(Record));
  S.num_tuples = static_cast<uint32_t>(t1.size());
  const uint32_t originalRSize = R.num_tuples;
  const uint32_t originalSSize = S.num_tuples;

  t0.clear();
  t0.shrink_to_fit();
  t1.clear();
  t1.shrink_to_fit();

  auto slices_R = buildSlices(R.num_tuples, thrR);
  auto slices_S = buildSlices(S.num_tuples, thrS);

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
      dummy.cntExpand = 0;
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

  std::uint32_t m;

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
  bitonic_sort_(R.tuples, true, 0, R.num_tuples, numThreads, false);
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

  std::thread partitionR([&] {
    std::vector<int> lastLen(slices_R.size()), mergeVal(slices_R.size() - 1);
    parallelCounts(R, slices_R, lastLen, mergeVal);
    replaceWithDummiesParallel(R, slices_R, SECRET);
  });
  std::thread partitionS([&] {
    std::vector<int> lastLen(slices_S.size()), mergeVal(slices_S.size() - 1);
    parallelCounts(S, slices_S, lastLen, mergeVal);
    replaceWithDummiesParallel(S, slices_S, SECRET);
  });
  partitionR.join();
  partitionS.join();

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
  assign_indices_parallel(R, numThreads);
  shuffleTable(S, numThreads);
  assign_indices_parallel(S, numThreads);
  std::chrono::high_resolution_clock::time_point tShuffleEnd =
      std::chrono::high_resolution_clock::now();

  std::chrono::high_resolution_clock::time_point onStart =
      std::chrono::high_resolution_clock::now();
  RHO(&R, &S, numThreads, bins);
  std::chrono::high_resolution_clock::time_point exchangeEnd =
      std::chrono::high_resolution_clock::now();

  auto cmp = [](const row_t &a, const row_t &b) { return a.idx < b.idx; };
  tbb::parallel_sort(R.tuples, R.tuples + R.num_tuples, cmp);
  tbb::parallel_sort(S.tuples, S.tuples + S.num_tuples, cmp);

  std::chrono::high_resolution_clock::time_point nonOblSortEnd =
      std::chrono::high_resolution_clock::now();

  R.num_tuples = originalRSize;
  S.num_tuples = originalSSize;

  std::thread processR([&] {
    backfillDummiesParallel(R, slices_R);
    auto selected = std::make_unique<bool[]>(R.num_tuples);
    m = prefixSumExpandParallel(R, slices_R, selected.get());
    obli_compact_rows(R.tuples, selected.get(), R.num_tuples, thrR);
    padTableToSize(R, m);
    obli_distribute_rows(R.tuples, m, numThreads / 2);
    carryForwardParallel(R, buildSlices(m, numThreads / 2));
  });
  std::thread processS([&] {
    backfillDummiesParallel(S, slices_S);
    auto selected = std::make_unique<bool[]>(S.num_tuples);
    m = prefixSumExpandParallel(S, slices_S, selected.get());
    obli_compact_rows(S.tuples, selected.get(), S.num_tuples, thrS);
    padTableToSize(S, m);
    obli_distribute_rows(S.tuples, m, numThreads / 2);
    carryForwardParallel(S, buildSlices(m, numThreads / 2));
  });
  processR.join();
  processS.join();

  std::chrono::high_resolution_clock::time_point distEnd =
      std::chrono::high_resolution_clock::now();

  alignTableParallel(S, buildSlices(m, numThreads), numThreads);
  std::chrono::high_resolution_clock::time_point alignEnd =
      std::chrono::high_resolution_clock::now();
  std::vector<JoinRec> joinResults;
  mergeExpandedParallel(R, S, numThreads, joinResults);

  double sec =
      std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart)
          .count();
  printf("\nJoin completed in %f s\n", sec);

  double onSec =
      std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - onStart)
          .count();
  printf("\nOnline: %f s\n", onSec);
  printf("\nOffline: %f s\n", sec - onSec);

  double exchangeSec =
      std::chrono::duration_cast<std::chrono::duration<double>>(exchangeEnd -
                                                                onStart)
          .count();
  printf("\nPartitioning and exchanging counts took %f s (%.2f%% of total "
         "execution time)\n",
         exchangeSec, (exchangeSec * 100.0 / sec));

  double nonOblSortSec =
      std::chrono::duration_cast<std::chrono::duration<double>>(nonOblSortEnd -
                                                                exchangeEnd)
          .count();
  printf("\nNon-oblivious sort took %f s (%.2f%% of online "
         "execution time)\n",
         nonOblSortSec, (nonOblSortSec * 100.0 / onSec));

  double distSec = std::chrono::duration_cast<std::chrono::duration<double>>(
                       distEnd - nonOblSortEnd)
                       .count();
  printf("\nBackfill dummies, Prefix calculation, and Dist & Expand took %f s "
         "(%.2f%% of online "
         "execution time)\n",
         distSec, (distSec * 100.0 / onSec));

  double alignSec = std::chrono::duration_cast<std::chrono::duration<double>>(
                        alignEnd - distEnd)
                        .count();
  printf("\nAlignment took %f s (%.2f%% of online "
         "execution time)\n",
         alignSec, (alignSec * 100.0 / onSec));

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
  printf("\nOShuffle took %f s (%.2f%% of total execution time)\n", shuffleSec,
         (shuffleSec * 100.0 / sec));

  // {
  //   std::ofstream outER("join.txt");
  //   for (const auto &j : joinResults)
  //     outER << j.keyR << ' ' << j.payR << ' ' << j.keyS << ' ' << j.payS
  //           << '\n';
  // }
  printf("Join result rows: %d (written to join.txt)\n", m);

  return 0;
}
