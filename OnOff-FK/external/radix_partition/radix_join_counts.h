#ifndef _RADIX_JOIN_COUNTS_H_
#define _RADIX_JOIN_COUNTS_H_

#include "data-types.h"
#include <stdbool.h>
#include <stdlib.h>

typedef int64_t (*JoinFunction)(const struct table_t *const,
                                const struct table_t *const,
                                struct table_t *const, output_list_t **output,
                                bool isSPrimary, int bins);

result_t *RHO(struct table_t *relR, struct table_t *relS, int nthreads,
              bool isSPrimary, int bins);

#endif //_RADIX_JOIN_COUNTS_H_
