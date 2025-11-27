#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <sys/random.h>
#include <tbb/global_control.h>
#include <tbb/parallel_sort.h>

#include "align_table.h"
#include "backfill_dummies.h"
#include "carry_forward.h"
#include "external_memory/algorithm/kway_butterfly_sort.hpp"
#include "inputs.h"
#include "merge.h"
#include "oblivious_ops.h"
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

// Global timer
std::chrono::high_resolution_clock::time_point tEnd;
std::uint32_t SECRET;

inline std::uint32_t prev_power_of_two(std::uint32_t n) {
  if (n <= 1)
    return 1;
  const unsigned leading = static_cast<unsigned>(__builtin_clz(n));
  return 1u << (31u - leading);
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
    std::cerr << "[WARNING] Lemma 1 unsatisfied. Reconfigure radix parameters."
              << std::endl;
  }

  return {prev_power_of_two(static_cast<std::uint32_t>(std::ceil(m))), p};
}

static void shuffleTable(table_t &tbl) {
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
        EM::Algorithm::KWayButterflyOShuffle(vec.begin(), vec.end(), 0,
                                             heapSize);
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

int main(int argc, char *argv[]) {
  printf("[INFO] Set number of radix bits and passes in the top-level "
         "CMakeLists.txt.\n");
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

  {
    ssize_t tmp = getrandom(&SECRET, sizeof(SECRET), 0);
    if (tmp != sizeof(SECRET)) {
      perror("Secret generation failed");
      exit(1);
    }
  }

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
  auto [bins, p] = findMaxBins(R.num_tuples / std::pow(2, NUM_RADIX_BITS));
  printf("Bins: %u, Lemma 1 p: %.4f\n", bins, p);

  std::chrono::high_resolution_clock::time_point tStart =
      std::chrono::high_resolution_clock::now();

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

  std::chrono::high_resolution_clock::time_point tShuffleStart =
      std::chrono::high_resolution_clock::now();
  shuffleTable(R);
  assign_indices_parallel(R, numThreads);
  std::chrono::high_resolution_clock::time_point tShuffleEnd =
      std::chrono::high_resolution_clock::now();

  RHO(&R, &S, numThreads, bins);

  auto cmp = [](const row_t &a, const row_t &b) { return a.idx < b.idx; };
  tbb::parallel_sort(R.tuples, R.tuples + R.num_tuples, cmp);

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

  alignTableParallel(S, buildSlices(m, numThreads), numThreads);
  std::vector<JoinRec> joinResults;
  mergeExpandedParallel(R, S, numThreads, joinResults);

  double sec =
      std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart)
          .count();
  printf("\nJoin completed in %f s\n", sec);

  double shuffleSec = std::chrono::duration_cast<std::chrono::duration<double>>(
                          tShuffleEnd - tShuffleStart)
                          .count();
  printf("\nOShuffle took %f s (%.2f%% of total execution time)\n", shuffleSec,
         (shuffleSec * 100.0 / sec));

  {
    std::ofstream outER("join.txt");
    for (const auto &j : joinResults)
      outER << j.keyR << ' ' << j.payR << ' ' << j.keyS << ' ' << j.payS
            << '\n';
  }
  printf("Join result rows: %d (written to join.txt)\n", m);

  return 0;
}
