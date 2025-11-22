#pragma once

#include <cstdint>

#include "external/radix_partition/data-types.h"

// Compacts rows in-place so that all entries with selected[i]==true move to the
// front while preserving their relative order. Length must equal the size of
// the selection bitmap. numThreads>=1 controls the amount of parallelism.
void obli_compact_rows(row_t *rows, const bool *selected, std::uint32_t length,
                       std::uint32_t numThreads);

// Inverse of obli_compact_rows. Rows are assumed to have valid idx fields
// identifying their target slot in [0, length). Each row with cntExpand>0 is
// placed at idx and the remaining gaps keep their dummy contents.
void obli_distribute_rows(row_t *rows, std::uint32_t length,
                          std::uint32_t numThreads);
