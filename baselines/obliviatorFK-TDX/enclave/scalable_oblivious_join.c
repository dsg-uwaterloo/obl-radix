#include "enclave/scalable_oblivious_join.h"
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <threads.h>
#include <sys/time.h>
#include <liboblivious/algorithms.h>
#include <liboblivious/primitives.h>
#include "common/elem_t.h"
#include "common/error.h"
#include "common/util.h"
#include "common/ocalls.h"
#include "common/code_conf.h"
//#include "enclave/mpi_tls.h"
#include "enclave/bitonic.h"
#include "enclave/parallel_enc.h"
#include "enclave/threading.h"
#include "enclave/oblivious_compact.h"

#ifndef DISTRIBUTED_SGX_SORT_HOSTONLY
#include <openenclave/enclave.h>
#include "enclave/parallel_t.h"
#endif

// Timing variables
static struct timeval start_time, end_time;
static double execution_time;

bool* control_bit;

static long long number_threads;

long long tree_node_idx_48[48] = {63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62};
long long tree_node_idx_6[6] = {7, 8, 9, 10, 5, 6};

void reverse(char *s) {
    long long i, j;
    char c;

    for (i = 0, j = strlen(s)-1; i<j; i++, j--) {
        c = s[i];
        s[i] = s[j];
        s[j] = c;
    }
}

long long my_len(char *data) {
    long long i = 0;
    while ((data[i] != '\0') && (i < DATA_LENGTH)) i++;
    
    return i;
}

long long o_strcmp(char* str1, char* str2) {
    bool flag = false;
    long long result = 0;

    for (long long i = 0; i < 105; i++) {
        result = !flag * (-1 * (str1[i] < str2[i]) + 1 * (str1[i] > str2[i])) + flag * result;
        flag = (!flag && ((str1[i] < str2[i]) || (str1[i] > str2[i]))) || flag;
    }

    return result;
}

void itoa(long long n, char *s, long long *len) {
    long long i = 0;
    long long sign;

    if ((sign = n) < 0) {
        n = -n;
        i = 0;
    }
        
    do {
        s[i++] = n % 10 + '0';
    } while ((n /= 10) > 0);
    
    if (sign < 0)
        s[i++] = '-';
    s[i] = '\0';
    
    *len = i;
    
    reverse(s);
}

void itoa2(int n, char *s, int *len) {
    int i = 0;
    int sign;
    if ((sign = n) < 0) {
        n = -n;
        i = 0;
    }
    do {
        s[i++] = n % 10 + '0';
    } while ((n /= 10) > 0);
    if (sign < 0)
        s[i++] = '-';
    s[i] = '\0';
    *len = i;
    reverse(s);
}


int scalable_oblivious_join_init(int nthreads) {
    number_threads = nthreads;
    return 0;
}

void scalable_oblivious_join_free() {
    return;
}



struct tree_node_op2* ag_tree;

void aggregation_tree_op2(void *voidargs) {
    struct args_op2 *args = (struct args_op2*) voidargs;
    long long index_thread_start = args->index_thread_start;
    long long index_thread_end = args->index_thread_end;
    elem_t* arr = args->arr;
    elem_t* arr_ = args->arr_;
    long long thread_order = args->thread_order;
    long long cur_tree_node = thread_order;
    bool condition;
    bool condition2;

    elem_t* arr_temp = calloc(1, sizeof(*arr_temp));
    arr_temp[0].table_0 = false;
    condition = arr[index_thread_start].table_0;
    o_memcpy(arr_temp, arr + index_thread_start, sizeof(*arr), condition);
    for (long long i = index_thread_start + 1; i < index_thread_end; i++) {
        condition = arr[i].table_0;
        condition2 = arr[i].key == arr_temp[0].key;
        o_memcpy(arr_ + i, arr_temp, sizeof(*arr_), ((!condition)&&condition2));
        o_memcpy(arr_temp, arr + i, sizeof(*arr), condition);
    }

    ag_tree[cur_tree_node].key_last = arr_temp[0].key;

    ag_tree[cur_tree_node].table0_last = arr_temp[0].table_0;
    ag_tree[cur_tree_node].complete1 = true;

    long long temp;
    while(cur_tree_node % 2 == 0 && 0 < cur_tree_node) {
        temp = (cur_tree_node - 2) / 2;
        while(!ag_tree[cur_tree_node - 1].complete1) {
            ;
        };
        condition = ag_tree[cur_tree_node].table0_last;
        ag_tree[temp].key_last = condition * ag_tree[cur_tree_node].key_last + !condition * ag_tree[cur_tree_node - 1].key_last;
        ag_tree[temp].table0_last = condition + ag_tree[cur_tree_node - 1].table0_last;
        ag_tree[temp].complete1 = true;
        cur_tree_node = temp;
    }
    long long temp1;

    while(cur_tree_node < thread_order) {
        temp = cur_tree_node * 2 + 2;
        temp1 = cur_tree_node * 2 + 1;
        while(!ag_tree[cur_tree_node].complete2) {
            ;
        };
        condition = 

        ag_tree[temp1].table0_prefix = ag_tree[cur_tree_node].table0_prefix;
        ag_tree[temp].table0_prefix = ag_tree[temp1].table0_last + ag_tree[cur_tree_node].table0_prefix;
            ag_tree[temp1].key_prefix = ag_tree[cur_tree_node].key_prefix;
            ag_tree[temp].key_prefix = ag_tree[temp1].table0_last * ag_tree[temp1].key_last + !ag_tree[temp1].table0_last * ag_tree[cur_tree_node].key_prefix;
        ag_tree[temp1].complete2 = true;
        ag_tree[temp].complete2 = true;
        cur_tree_node = temp;
    }
    while(!ag_tree[thread_order].complete2) {
        ;
    };

        arr_temp[0].key = ag_tree[thread_order].key_prefix;
    arr_temp[0].table_0 = ag_tree[thread_order].table0_prefix;

    for (long long i = index_thread_start; i < index_thread_end; i++) {
        condition = !arr[i].table_0 && !arr_[i].table_0;
        condition2 = (arr[i].key == arr_temp[0].key);
        o_memcpy(arr_ + i, arr_temp, sizeof(*arr_temp), condition && condition2);
        control_bit[i] = !arr[i].table_0 && arr_[i].table_0;
    }

    free(arr_temp);
    return;
}

void scalable_oblivious_join(elem_t *arr, long long length1, long long length2, char* output_path){
    long long length = length1 + length2;
    elem_t* arr_ = calloc(length, sizeof(*arr_));
    for (long long i = 0; i < length; i++) {
        arr_[i].table_0 = false;
    }
    bool condition;
    bool condition2;
    long long length_thread = length / number_threads;
    long long length_extra = length % number_threads;
    struct args_op2 args_op2_[number_threads];
    long long idx_start_thread[number_threads + 1];
    idx_start_thread[0] = 0;
    struct thread_work multi_thread_aggregation_tree_1[number_threads - 1];
    ag_tree = calloc(2 * number_threads - 1, sizeof(*ag_tree));
    ag_tree[0].table0_prefix = 0;
    ag_tree[0].complete2 = true;
    elem_t* arr_temp = calloc(1, sizeof(*arr_temp));
    control_bit = calloc(length, sizeof(*control_bit));
    int result_length = 0;
    control_bit[0] = false;
    // init_time2();
    gettimeofday(&start_time, NULL);

    bitonic_sort(arr, true, 0, length, number_threads, true);

    if (number_threads == 1) {
        condition = arr[0].table_0;
        o_memcpy(arr_temp, arr, sizeof(*arr), condition);
        for (long long i = 1; i < length; i++) {
            condition = arr[i].table_0;
            condition2 = arr[i].key == arr_temp[0].key;
            control_bit[i] = (!condition) && condition2; 
            o_memcpy(arr_ + i, arr_temp, sizeof(*arr_), control_bit[i]);
            o_memcpy(arr_temp, arr + i, sizeof(*arr), condition);
        }
    } else {
        for (long long i = 0; i < number_threads; i++) {
            idx_start_thread[i + 1] = idx_start_thread[i] + length_thread + (i < length_extra);

            args_op2_[i].arr = arr;
                args_op2_[i].arr_ = arr_;
                args_op2_[i].index_thread_start = idx_start_thread[i];
                args_op2_[i].index_thread_end = idx_start_thread[i + 1];
                if (number_threads == 6) {
                    args_op2_[i].thread_order = tree_node_idx_6[i];        
                } else if (number_threads == 48) {
                    args_op2_[i].thread_order = tree_node_idx_48[i];
                } else {
                    args_op2_[i].thread_order = number_threads + i - 1;
                }
                if (i < number_threads - 1) {
                    multi_thread_aggregation_tree_1[i].type = THREAD_WORK_SINGLE;
                    multi_thread_aggregation_tree_1[i].single.func = aggregation_tree_op2;
                    multi_thread_aggregation_tree_1[i].single.arg = &args_op2_[i];
                    thread_work_push(&multi_thread_aggregation_tree_1[i]);
                }
        }
        aggregation_tree_op2(&args_op2_[number_threads - 1]);
        for (long long i = 0; i < number_threads - 1; i++) {
            thread_wait(&multi_thread_aggregation_tree_1[i]);
        }
    }

    result_length = oblivious_compact_elem(arr, control_bit, length, 1, number_threads);
    oblivious_compact_elem(arr_, control_bit, length, 1, number_threads);
    
    // get_time2(true);
    // End timing
    gettimeofday(&end_time, NULL);
    execution_time = (end_time.tv_sec - start_time.tv_sec) + 
                       (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
    printf("Execution time: %.6f seconds\n", execution_time);
    printf("length_result: %d\n", result_length);
    
    char *char_current = output_path;
    for (int i = 0; i < result_length; i++) {
        int key1 = arr[i].key;
        int key2 = arr_[i].key;

        char string_key1[10];
        char string_key2[10];
        int str1_len, str2_len;
        itoa2(key1, string_key1, &str1_len);
        itoa2(key2, string_key2, &str2_len);
        int data_len1 = my_len(arr[i].data);
        int data_len2 = my_len(arr_[i].data);
        
        strncpy(char_current, string_key1, str1_len);
        char_current += str1_len; char_current[0] = ' '; char_current += 1;
        strncpy(char_current, arr[i].data, data_len1);
        char_current += data_len1; char_current[0] = ' '; char_current += 1;
        strncpy(char_current, string_key2, str2_len);
        char_current += str2_len; char_current[0] = ' '; char_current += 1;
        strncpy(char_current, arr_[i].data, data_len2);
        char_current += data_len2; char_current[0] = '\n'; char_current += 1;
    }
    char_current[0] = '\0';

    free(ag_tree);
    free(arr_temp);
    free(arr_);
    free(control_bit);
    
    return;
}