#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>

#include "generate_hash.h"
#include "inputs.h"
#include "mark_expand.h"
#include "oblivious_ops.h"
#include "slice_utils.h"

extern "C" {
#include "radix_join_counts.h"
}


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

static void printOutput(const char *label, const table_t &tbl) {
  for (uint64_t i = 0; i < tbl.num_tuples; ++i) {
    const row_t &rec = tbl.tuples[i];
    printf("key=%u cntSelf=%u idx=%u hashKey=%u self=%s primary=%s\n", rec.key,
           rec.cntSelf, rec.idx, rec.hashKey, rec.paySelf, rec.payPrimary);
  }
}


int main(int argc, char *argv[]) {
  std::chrono::high_resolution_clock::time_point tStart, tEnd;
  printf("[INFO] Set number of radix bits and passes in the top-level "
         "CMakeLists.txt.\n");
  printf("[INFO] R: Primary Key table; S: Foreign Key table\n");
  std::uint32_t numThreads = 32;
  std::string inputPath = "../../datasets/real/imdb/imdb.txt"; // UPDATE

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

  std::vector<Record> t0, t1;
  if (!load_two_tables(inputPath, t0, t1))
    return 1;

  bool swapped = false;
  if (t0.size() > t1.size()) {
    std::swap(t0, t1);
    swapped = true;
  }

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

  std::uint32_t thrR = std::max<std::uint32_t>(
      1, ceil((static_cast<double>(t0.size()) / (t0.size() + t1.size())) *
              numThreads));
  std::uint32_t thrS = std::max<std::uint32_t>(1, numThreads - thrR);
  printf("threads_R: %u, threads_S: %u\n", thrR, thrS);

  auto slices_R_numThreads = buildSlices(R.num_tuples, thrR);
  auto slices_S_numThreads = buildSlices(S.num_tuples, thrS);
  auto slices_final = buildSlices(R.num_tuples, numThreads);

  printf("\nRadix bits: %u, Passes: %u\n", NUM_RADIX_BITS, NUM_PASSES);
  std::uint32_t bins;

  double p;
  if (R.num_tuples <= S.num_tuples) {
    std::tie(bins, p) = findMaxBins(R.num_tuples / std::pow(2, NUM_RADIX_BITS));
  } else {
    std::tie(bins, p) = findMaxBins(S.num_tuples / std::pow(2, NUM_RADIX_BITS));
  }
  printf("Bins: %u, Lemma 1 p: %.4f\n", bins, p);

  tStart = std::chrono::high_resolution_clock::now();

  std::thread hashR([&] { generateHashParallel(R, slices_R_numThreads); });
  std::thread hashS([&] { generateHashParallel(S, slices_S_numThreads); });
  hashR.join();
  hashS.join();

  RHO(&R, &S, numThreads, bins);

  auto selected = std::make_unique<bool[]>(R.num_tuples);
  std::uint32_t m = markExpandParallel(R, slices_final, selected.get());
  if (m != R.num_tuples)
    obli_compact_rows(R.tuples, selected.get(), R.num_tuples, numThreads);
  tEnd = std::chrono::high_resolution_clock::now();

  double sec =
      std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart)
          .count();
  printf("\nJoin completed in %f s\n", sec);
  {
    std::ofstream outER("join.txt");
    for (int i = 0; i < m; i++) {
      if (!swapped) {
        outER << R.tuples[i].key << ' ' << R.tuples[i].paySelf << ' '
              << R.tuples[i].key << ' ' << R.tuples[i].payPrimary << '\n';
      } else {
        outER << R.tuples[i].key << ' ' << R.tuples[i].payPrimary << ' '
              << R.tuples[i].key << ' ' << R.tuples[i].paySelf << '\n';
      }
    }
  }
  printf("Join result rows: %d (written to join.txt)\n", m);

  return 0;
}
