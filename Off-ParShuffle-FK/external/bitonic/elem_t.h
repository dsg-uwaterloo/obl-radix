#pragma once

#ifdef __cplusplus
#include "../radix_partition/data-types.h"  
typedef row_t elem_t;
#else
#include <stdint.h>

#ifndef DATA_LENGTH
#define DATA_LENGTH 4
#endif

typedef struct {
    uint32_t key;
    uint32_t cntSelf;
    uint32_t hashKey;
    uint32_t idx;
    uint32_t shuffledIdx;
    char paySelf[DATA_LENGTH];
    char payPrimary[DATA_LENGTH];

} __attribute__((aligned(32))) elem_t;

#define j_order idx  
#endif
