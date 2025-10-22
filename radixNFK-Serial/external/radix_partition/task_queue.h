/**
 * @file    task_queue.h
 * @author  Cagri Balkesen <cagri.balkesen@inf.ethz.ch>
 * @date    Sat Feb  4 20:00:58 2012
 * @version $Id: task_queue.h 3017 2012-12-07 10:56:20Z bcagri $
 * 
 * @brief  Implements task queue facility for the join processing.
 * 
 */
#ifndef TASK_QUEUE_H
#define TASK_QUEUE_H

#include <pthread.h>
#include <stdlib.h>

#include "data-types.h" 

/** 
 * @defgroup TaskQueue Task Queue Implementation 
 * @{
 */

typedef struct task_t task_t;
typedef struct task_list_t task_list_t;
typedef struct task_queue_t task_queue_t;

struct task_t {
    struct table_t relR;
    struct table_t tmpR;
    struct table_t relS;
    struct table_t tmpS;
    task_t *   next;
};

struct task_list_t {
    task_t *      tasks;
    task_list_t * next;
    int           curr;
};

struct task_queue_t {
    pthread_mutex_t lock;
    pthread_mutex_t alloc_lock;
    task_t *        head;
    task_list_t *   free_list;
    int32_t         count;
    int32_t         alloc_size;
};

inline 
task_t * 
get_next_task(task_queue_t * tq) __attribute__((always_inline));

inline 
void 
add_tasks(task_queue_t * tq, task_t * t) __attribute__((always_inline));

inline 
task_t * 
get_next_task(task_queue_t * tq) 
{
    pthread_mutex_lock(&tq->lock);
    task_t * ret = 0;
    if(tq->count > 0){
        ret = tq->head;
        tq->head = ret->next;
        tq->count --;
    }
    pthread_mutex_unlock(&tq->lock);

    return ret;
}

inline 
void 
add_tasks(task_queue_t * tq, task_t * t) 
{
    pthread_mutex_lock(&tq->lock);
    t->next = tq->head;
    tq->head = t;
    tq->count ++;
    pthread_mutex_unlock(&tq->lock);
}

/* atomically get the next available task */
task_t * 
task_queue_get_atomic(task_queue_t * tq);


/* atomically add a task */
void 
task_queue_add_atomic(task_queue_t * tq, task_t * t);

void 
task_queue_add(task_queue_t * tq, task_t * t);

void 
task_queue_copy_atomic(task_queue_t * tq, task_t * t);

/* get a free slot of task_t */
task_t * 
task_queue_get_slot_atomic(task_queue_t * tq);

task_t * 
task_queue_get_slot(task_queue_t * tq);


/* initialize a task queue with given allocation block size */
task_queue_t * 
task_queue_init(int alloc_size);

void 
task_queue_free(task_queue_t * tq);

#endif /* TASK_QUEUE_H */
