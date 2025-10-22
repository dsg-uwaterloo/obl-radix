#ifndef _RADIX_JOIN_IDX_H_
#define _RADIX_JOIN_IDX_H_

#include "data-types.h"
#include <stdbool.h>
#include <stdlib.h>

typedef int64_t (*JoinFunctionIdx)(const struct table_t *const,
                                   const struct table_t *const,
                                   struct table_t *const,
                                   output_list_t **output,
                                   struct table_t *expanded, bool isIdxS);

result_t *RHO_idx(struct table_t *relR, struct table_t *relS, int nthreads,
                  struct table_t *expanded, bool isIdxS);

#endif //_RADIX_JOIN_IDX_H_
