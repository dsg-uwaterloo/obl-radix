#ifndef DISTRIBUTED_SGX_SORT_ENCLAVE_BITONIC_H
#define DISTRIBUTED_SGX_SORT_ENCLAVE_BITONIC_H

#include <stdbool.h>
#include <stddef.h>
//#include "common/defs.h"
#include "elem_t.h"

struct bitonic_sort_new_args {
    bool ascend;
    int lo;
    int hi;
    int number_threads;
};

void bitonic_sort_(elem_t *arr_, bool ascend , int lo, int hi, int num_threads, bool D2enable);

// Sorts by the lowest `total_bits` bits of `hashKey` (ascending), then by `idx`
// (ascending). Intended to group by (partition,bin) while sinking dummies with
// idx==UINT32_MAX.
void bitonic_sort_hashkey_lowbits_(elem_t *arr_, bool ascend, int lo, int hi,
                                  int num_threads, unsigned total_bits);

#endif /* distributed-sgx-sort/enclave/bitonic.h */
