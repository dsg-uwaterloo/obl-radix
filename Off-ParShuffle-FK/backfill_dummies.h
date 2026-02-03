#pragma once
#include "data-types.h"
#include "inputs.h"
#include "slice_utils.h"
#include <immintrin.h>
#include <limits>
#include <thread>
#include <vector>

using std::vector;

inline void backfillDummiesParallel(table_t &tbl,
                                    const std::vector<Slice> &slices) {
  using u32_may_alias = std::uint32_t __attribute__((__may_alias__));

  const std::size_t P = slices.size();
  if (P == 0)
    return;

  std::vector<Record> tail(P);

  std::vector<std::thread> pool;
  pool.reserve(P);
  for (std::size_t t = 0; t < P; ++t) {
    pool.emplace_back([&, t] {
      const Slice sl = slices[t];
      Record last{};
      for (ssize_t i = static_cast<ssize_t>(sl.end) - 1;
           i >= static_cast<ssize_t>(sl.begin); --i) {
        if (i >= 16)
          _mm_prefetch(reinterpret_cast<const char *>(&tbl.tuples[i - 16]),
                       _MM_HINT_T0);
        std::uint64_t isReal = -(tbl.tuples[i].cntSelf != 0);
        maskedCopyRecord32(reinterpret_cast<Record *>(&tbl.tuples[i]), &last,
                           isReal);
      }
      tail[t] = last;
    });
  }
  for (auto &th : pool)
    th.join();

  std::vector<Record> seed(P);
  Record running{};
  for (ssize_t t = static_cast<ssize_t>(P) - 1; t >= 0; --t) {
    seed[t] = running;
    std::uint64_t hasReal = -(tail[t].cntSelf != 0);
    maskedCopyRecord32(reinterpret_cast<const Record *>(&tail[t]),
                       reinterpret_cast<Record *>(&running), hasReal);
  }

  pool.clear();
  pool.reserve(P);
  for (std::size_t t = 0; t < P; ++t) {
    pool.emplace_back([&, t] {
      const Slice sl = slices[t];
      std::uint32_t lastKey = seed[t].key;
      std::uint32_t lastSelf = seed[t].cntSelf;
      std::uint32_t lastPayPrimaryU32 = 0;
      std::memcpy(&lastPayPrimaryU32, seed[t].payPrimary, sizeof(lastPayPrimaryU32));

      for (ssize_t i = static_cast<ssize_t>(sl.end) - 1;
           i >= static_cast<ssize_t>(sl.begin); --i) {
        if (i >= 16)
          _mm_prefetch(reinterpret_cast<const char *>(&tbl.tuples[i - 16]),
                       _MM_HINT_T0);
        std::uint32_t isReal = -(tbl.tuples[i].cntSelf != 0);

        std::uint32_t newKey =
            (isReal & tbl.tuples[i].key) | (~isReal & lastKey);
        std::uint32_t newSelf =
            (isReal & tbl.tuples[i].cntSelf) | (~isReal & lastSelf);

        u32_may_alias *payP =
            reinterpret_cast<u32_may_alias *>(&tbl.tuples[i].payPrimary[0]);
        const std::uint32_t curPayPrimary = *payP;
        const std::uint32_t newPayPrimary =
            (isReal & curPayPrimary) | (~isReal & lastPayPrimaryU32);
        *payP = newPayPrimary;
        lastPayPrimaryU32 = newPayPrimary;

        tbl.tuples[i].key = newKey;
        tbl.tuples[i].cntSelf = newSelf;

        lastKey = newKey;
        lastSelf = newSelf;
      }
    });
  }
  for (auto &th : pool)
    th.join();
}
