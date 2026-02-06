#pragma once
#include <cstdint>
#include <cstdio>
#include <vector>

struct Slice {
  std::uint32_t begin, end;
};

inline std::vector<Slice> buildSlices(std::uint32_t N,
                                      std::uint32_t numThreads) {
  if (numThreads == 0)
    numThreads = 1;
  numThreads = std::min(numThreads, N);

  std::uint32_t base = N / numThreads;
  std::uint32_t extra = N % numThreads;

  std::vector<Slice> slices(numThreads);
  std::uint32_t start = 0;
  for (std::uint32_t t = 0; t < numThreads; ++t) {
    std::uint32_t len = base + (t < extra);
    slices[t] = {start, start + len};
    start += len;
  }
  return slices;
}
