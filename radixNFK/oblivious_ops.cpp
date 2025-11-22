#include "oblivious_ops.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <thread>
#include <vector>

#include "inputs.h"

namespace {

constexpr std::uint32_t kParallelThreshold = 1u << 12; // 4096 elements

inline std::uint32_t clamp_threads(std::uint32_t t) {
  return std::max<std::uint32_t>(1, t);
}

inline std::uint32_t prev_power_of_two(std::uint32_t n) {
  if (n <= 1)
    return 1;
  const unsigned leading = static_cast<unsigned>(__builtin_clz(n));
  return 1u << (31u - leading);
}

// inline void oblivious_swap(row_t &a, row_t &b, bool flag) {
//   const std::uint64_t mask =
//       flag ? std::numeric_limits<std::uint64_t>::max() : 0;
//   row_t tmp = a;
//   maskedCopyRecord32(reinterpret_cast<const Record *>(&b),
//                      reinterpret_cast<Record *>(&a), mask);
//   maskedCopyRecord32(reinterpret_cast<const Record *>(&tmp),
//                      reinterpret_cast<Record *>(&b), mask);
// }

inline void oblivious_swap(row_t &a, row_t &b, bool flag) {
  const std::uint64_t mask =
      static_cast<std::uint64_t>(0) - static_cast<std::uint64_t>(flag);
  row_t tmp = a;
  maskedCopyRecord32(reinterpret_cast<const Record *>(&b),
                     reinterpret_cast<Record *>(&a), mask);
  maskedCopyRecord32(reinterpret_cast<const Record *>(&tmp),
                     reinterpret_cast<Record *>(&b), mask);
}


template <class Func>
void parallel_for(std::uint32_t begin, std::uint32_t end,
                  std::uint32_t maxThreads, const Func &fn) {
  if (end <= begin)
    return;
  const std::uint32_t len = end - begin;
  if (maxThreads <= 1 || len < kParallelThreshold) {
    fn(begin, end);
    return;
  }
  const std::uint32_t usableThreads = std::min(maxThreads, len);
  std::vector<std::thread> workers;
  workers.reserve(usableThreads - 1);
  std::uint32_t cur = begin;
  const std::uint32_t baseChunk = len / usableThreads;
  const std::uint32_t extra = len % usableThreads;
  for (std::uint32_t t = 0; t < usableThreads; ++t) {
    const std::uint32_t next = cur + baseChunk + (t < extra);
    if (t + 1 == usableThreads) {
      fn(cur, next);
    } else {
      workers.emplace_back([cur, next, &fn]() { fn(cur, next); });
    }
    cur = next;
  }
  for (auto &th : workers)
    th.join();
}

// ---------- Oblivious compaction ----------

void compact_pow2(row_t *rows, const bool *selected,
                  const std::uint32_t *prefix, std::uint32_t length,
                  std::uint32_t offset, std::uint32_t threads);

void compact_generic(row_t *rows, const bool *selected,
                     const std::uint32_t *prefix, std::uint32_t length,
                     std::uint32_t threads) {
  if (length <= 1)
    return;
  if (length == 2) {
    const bool swap = (!selected[0] && selected[1]);
    oblivious_swap(rows[0], rows[1], swap);
    return;
  }

  const std::uint32_t pow2 = prev_power_of_two(length);
  const std::uint32_t remainder = length - pow2;
  if (remainder == 0) {
    compact_pow2(rows, selected, prefix, pow2, 0, threads);
    return;
  }

  const std::uint32_t markedR = prefix[remainder] - prefix[0];
  const bool enableParallel = (threads > 1 && length >= kParallelThreshold);
  std::uint32_t rightThreads = enableParallel ? threads * pow2 / length : 1;
  rightThreads = std::max<std::uint32_t>(1, rightThreads);
  std::uint32_t leftThreads =
      enableParallel ? std::max<std::uint32_t>(1, threads - rightThreads) : 1;

  const std::uint32_t offset =
      (pow2 - remainder + markedR) & (pow2 - 1); // safe because pow2 power of 2

  std::thread rightWorker;
  if (enableParallel && rightThreads > 1) {
    rightWorker = std::thread([=]() {
      compact_pow2(rows + remainder, selected + remainder, prefix + remainder,
                   pow2, offset, rightThreads);
    });
  } else {
    compact_pow2(rows + remainder, selected + remainder, prefix + remainder,
                 pow2, offset, rightThreads);
  }
  compact_generic(rows, selected, prefix, remainder, leftThreads);
  if (rightWorker.joinable())
    rightWorker.join();

  auto finalSwap = [&](std::uint32_t begin, std::uint32_t end) {
    for (std::uint32_t i = begin; i < end; ++i) {
      const bool swap = (i >= markedR);
      oblivious_swap(rows[i], rows[i + pow2], swap);
    }
  };
  parallel_for(0, remainder, enableParallel ? threads : 1, finalSwap);
}

void compact_pow2(row_t *rows, const bool *selected,
                  const std::uint32_t *prefix, std::uint32_t length,
                  std::uint32_t offset, std::uint32_t threads) {
  if (length <= 1)
    return;
  if (length == 2) {
    const bool swap = ((!selected[0] && selected[1]) ^
                       static_cast<bool>(offset & 1u));
    oblivious_swap(rows[0], rows[1], swap);
    return;
  }

  const std::uint32_t half = length / 2;
  const std::uint32_t mLeft = prefix[half] - prefix[0];
  const std::uint32_t offsetMod = offset & (half - 1);
  const std::uint32_t offsetMLeft = (offset + mLeft) & (half - 1);
  const bool offsetRight = offset >= half;
  const bool leftWrapped = (offsetMod + mLeft) >= half;
  const bool enableParallel = (threads > 1 && length >= kParallelThreshold);
  std::uint32_t rightThreads =
      enableParallel ? std::max<std::uint32_t>(1, threads - threads / 2) : 1;
  std::uint32_t leftThreads =
      enableParallel ? std::max<std::uint32_t>(1, threads / 2) : 1;

  std::thread rightWorker;
  if (enableParallel && rightThreads > 1) {
    rightWorker = std::thread([=]() {
      compact_pow2(rows + half, selected + half, prefix + half, half,
                   offsetMLeft, rightThreads);
    });
  } else {
    compact_pow2(rows + half, selected + half, prefix + half, half, offsetMLeft,
                 rightThreads);
  }
  compact_pow2(rows, selected, prefix, half, offsetMod, leftThreads);
  if (rightWorker.joinable())
    rightWorker.join();

  auto swapWorker = [&](std::uint32_t begin, std::uint32_t end) {
    for (std::uint32_t i = begin; i < end; ++i) {
      const bool swap = (i >= offsetMLeft) ^ leftWrapped ^ offsetRight;
      oblivious_swap(rows[i], rows[i + half], swap);
    }
  };
  parallel_for(0, half, enableParallel ? threads : 1, swapWorker);
}

// ---------- Oblivious expansion ----------

inline bool row_has_value(const row_t &row) { return row.cntExpand > 0; }

void distribute_pow2(row_t *rows, std::uint32_t indexStart, std::uint32_t offset,
                     std::uint32_t length, std::uint32_t threads);

void distribute_generic(row_t *rows, std::uint32_t indexStart,
                        std::uint32_t length, std::uint32_t threads) {
  if (length <= 1)
    return;
  if (length == 2) {
    const bool swap = (row_has_value(rows[0]) && (rows[0].idx >= indexStart + 1)) ||
                      (row_has_value(rows[1]) && (rows[1].idx < indexStart + 1));
    oblivious_swap(rows[0], rows[1], swap);
    return;
  }

  const std::uint32_t pow2 = prev_power_of_two(length);
  const std::uint32_t remainder = length - pow2;
  if (remainder == 0) {
    distribute_pow2(rows, indexStart, 0, pow2, threads);
    return;
  }

  std::uint32_t m2 = 0;
  const std::uint32_t pivot = indexStart + remainder;
  for (std::uint32_t i = 0; i < remainder; ++i)
    m2 += (rows[i].idx < pivot);

  auto swapStage = [&](std::uint32_t begin, std::uint32_t end) {
    for (std::uint32_t i = begin; i < end; ++i) {
      const bool swap = (m2 <= i);
      oblivious_swap(rows[i], rows[i + pow2], swap);
    }
  };
  const bool enableParallel = (threads > 1 && length >= kParallelThreshold);
  parallel_for(0, remainder, enableParallel ? threads : 1, swapStage);

  std::uint32_t rightThreads = enableParallel ? threads * pow2 / length : 1;
  rightThreads = std::max<std::uint32_t>(1, rightThreads);
  std::uint32_t leftThreads =
      enableParallel ? std::max<std::uint32_t>(1, threads - rightThreads) : 1;
  const std::uint32_t offset =
      (pow2 - remainder + m2) & (pow2 - 1); // modulo pow2

  std::thread rightWorker;
  if (enableParallel && rightThreads > 1) {
    rightWorker = std::thread([=]() {
      distribute_pow2(rows + remainder, indexStart + remainder, offset, pow2,
                      rightThreads);
    });
  } else {
    distribute_pow2(rows + remainder, indexStart + remainder, offset, pow2,
                    rightThreads);
  }
  distribute_generic(rows, indexStart, remainder, leftThreads);
  if (rightWorker.joinable())
    rightWorker.join();
}

void distribute_pow2(row_t *rows, std::uint32_t indexStart, std::uint32_t offset,
                     std::uint32_t length, std::uint32_t threads) {
  if (length <= 1)
    return;
  if (length == 2) {
    const bool swap =
        (row_has_value(rows[0]) && (rows[0].idx >= indexStart + 1)) ||
        (row_has_value(rows[1]) && (rows[1].idx < indexStart + 1));
    oblivious_swap(rows[0], rows[1], swap);
    return;
  }

  const std::uint32_t half = length / 2;
  const std::uint32_t pivot = indexStart + half;
  std::uint32_t m2 = 0;
  for (std::uint32_t i = 0; i < length; ++i)
    m2 += (rows[i].idx < pivot);

  const bool cond0 = (offset < half);
  const bool cond1 = (((offset % half) + m2) < half);
  const bool flag = cond0 ^ cond1;
  const std::uint32_t boundary = (offset + m2) % half;

  auto swapStage = [&](std::uint32_t begin, std::uint32_t end) {
    for (std::uint32_t i = begin; i < end; ++i) {
      const bool swap = flag ^ (i >= boundary);
      oblivious_swap(rows[i], rows[i + half], swap);
    }
  };
  const bool enableParallel = (threads > 1 && length >= kParallelThreshold);
  parallel_for(0, half, enableParallel ? threads : 1, swapStage);

  std::uint32_t rightThreads =
      enableParallel ? std::max<std::uint32_t>(1, threads - threads / 2) : 1;
  std::uint32_t leftThreads =
      enableParallel ? std::max<std::uint32_t>(1, threads / 2) : 1;

  std::thread rightWorker;
  if (enableParallel && rightThreads > 1) {
    rightWorker = std::thread([=]() {
      distribute_pow2(rows + half, indexStart + half, (offset + m2) % half, half,
                      rightThreads);
    });
  } else {
    distribute_pow2(rows + half, indexStart + half, (offset + m2) % half, half,
                    rightThreads);
  }
  distribute_pow2(rows, indexStart, offset % half, half, leftThreads);
  if (rightWorker.joinable())
    rightWorker.join();
}

} // namespace

void obli_compact_rows(row_t *rows, const bool *selected, std::uint32_t length,
                       std::uint32_t numThreads) {
  if (!rows || !selected || length <= 1)
    return;
  std::vector<std::uint32_t> prefix(length + 1, 0);
  for (std::uint32_t i = 0; i < length; ++i)
    prefix[i + 1] = prefix[i] + static_cast<std::uint32_t>(selected[i]);
  compact_generic(rows, selected, prefix.data(), length,
                  clamp_threads(numThreads));
}

void obli_distribute_rows(row_t *rows, std::uint32_t length,
                          std::uint32_t numThreads) {
  if (!rows || length <= 1)
    return;
  distribute_generic(rows, 0, length, clamp_threads(numThreads));
}
