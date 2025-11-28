#pragma once

#include "external/radix_partition/data-types.h"
#include <cstdint>

// Compacts rows in-place so that all entries with selected[i]==true move to the
// front while preserving their relative order. Length must equal the size of
// the selection bitmap. numThreads>=1 controls the amount of parallelism.
void obli_compact_rows(row_t *rows, const bool *selected, std::uint32_t length,
                       std::uint32_t numThreads);
