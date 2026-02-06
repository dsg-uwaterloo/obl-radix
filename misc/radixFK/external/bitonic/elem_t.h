#pragma once

#ifdef __cplusplus
#include "../radix_partition/data-types.h"  
typedef row_t elem_t;
#else
#include <stdint.h>

#ifndef DATA_LENGTH
#define DATA_LENGTH 8
#endif

typedef struct {
    uint32_t key;
    uint32_t cntSelf;      
    uint32_t hashKey;
    uint32_t idx; 
    char     paySelf[DATA_LENGTH];
    char     payPrimary[DATA_LENGTH];
} elem_t;

#define j_order idx  
#endif
