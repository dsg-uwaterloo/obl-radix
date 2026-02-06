#include "data-types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void malloc_check_ex(void *ptr, char const *file, char const *function,
                     int line) {
  if (ptr == NULL) {
    printf("%s:%s:%d Failed to allocate memory\n", file, function, line);
    exit(EXIT_FAILURE);
  }
}

void insert_output(output_list_t **head, type_key key, type_value Rpayload,
                   type_value Spayload) {
  output_list_t *row = (output_list_t *)malloc(sizeof(output_list_t));
  malloc_check_ex(row, __FILE__, __FUNCTION__, __LINE__);
  row->key = key;
  memcpy(row->Rpayload, Rpayload,
         sizeof(type_value)); 
  memcpy(row->Spayload, Spayload,
         sizeof(type_value));
  row->next = *head;
  *head = row;
}

uint64_t sizeof_output(output_list_t *list) {
  uint64_t size = 0;
  output_list_t *cur = list;
  while (cur != NULL) {
    size++;
    cur = cur->next;
  }
  return size * sizeof(output_list_t);
}

uint64_t sizeof_result(result_t *result) {
  uint64_t size = sizeof(result->totalresults) + sizeof(result->nthreads);
  for (int i = 0; i < result->nthreads; i++) {
    size += sizeof(result->resultlist[i].nresults);
    size += sizeof(result->resultlist[i].threadid);
    //        size += sizeof_output(result->resultlist[i].results);
    size += result->resultlist[i].nresults * sizeof(output_list_t);
  }
  return size;
}
