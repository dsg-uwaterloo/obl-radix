#include "radix_join_idx.h"
#include "barrier.h"
#include "data-types.h"
#include "malloc.h"
#include "prj_params.h"
#include "task_queue.h"
#include "util.h"
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>

#define HASH_BIT_MODULO(K, MASK, NBITS) (((K) & MASK) >> NBITS)
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

// inspired from "bit twiddling hacks":
// http://graphics.stanford.edu/~seander/bithacks.html
// #define PREV_POW_2(V)                                                          \
//   do {                                                                         \
//     V |= V >> 1;                                                               \
//     V |= V >> 2;                                                               \
//     V |= V >> 4;                                                               \
//     V |= V >> 8;                                                               \
//     V |= V >> 16;                                                              \
//     V = V - (V >> 1);                                                          \
//   } while (0)

typedef struct arg_t_radix arg_t_radix;
typedef struct part_t part_t;

/** holds the arguments passed to each thread */
struct arg_t_radix {
  int32_t **histR;
  struct row_t *relR;
  struct row_t *tmpR;
  int32_t **histS;
  struct row_t *relS;
  struct row_t *tmpS;

  /* Original array pointers for propagation */
  struct row_t *origRelR;
  struct row_t *origRelS;

  struct row_t *tmpR2;
  struct row_t *tmpS2;

  struct table_t *expanded_tbl;
  bool isIdxS;

  uint64_t numR;
  uint64_t numS;
  uint64_t totalR;
  uint64_t totalS;

  task_queue_t *join_queue;
  task_queue_t *part_queue;

  pthread_barrier_t *barrier;
  JoinFunctionIdx join_function;
  int64_t result;
  int32_t my_tid;
  int nthreads;

  /* stats about the thread */
  int32_t parts_processed;
  uint64_t timer1, timer2, timer3;
  uint64_t start, end;
  uint64_t pass1, pass2;
} __attribute__((aligned(CACHE_LINE_SIZE)));

/** holds arguments passed for partitioning */
struct part_t {
  struct row_t *rel;
  struct row_t *tmp;
  int32_t **hist;
  int32_t *output;
  arg_t_radix *thrargs;
  uint32_t num_tuples;
  uint32_t total_tuples;
  int32_t R;
  uint32_t D;
  int relidx; /* 0: R, 1: S */
  uint32_t padding;
} __attribute__((aligned(CACHE_LINE_SIZE)));

static void *alloc_aligned(size_t size) {
  void *ret;
  ret = memalign(CACHE_LINE_SIZE, size);

  malloc_check(ret);

  return ret;
}

/**
 * Find the maximum number of bins that achieves a target probability
 * using binary search to solve: m * exp(-n/m) ≈ target_p
 */
// static uint32_t findMaxBins(double n, double target_p, double eps) {
//   double low = 1, high = n, m = 0, p = 0;
//   for (int i = 0; i < 100; ++i) {
//     m = (low + high) / 2.0;
//     p = m * exp(-n / m);
//     if (fabs(p - target_p) < eps)
//       break;
//     (p > target_p) ? (high = m) : (low = m);
//   }
//   return ceil(m);
// }

int64_t bucket_chaining_join_idx(const struct table_t *const R,
                                 const struct table_t *const S,
                                 struct table_t *const tmpR,
                                 output_list_t **output,
                                 struct table_t *expanded, bool isIdxS) {
  (void)(tmpR);
  (void)(output);

  int *next, *bucket;
  const uint64_t numR = R->num_tuples;
  const uint64_t numS = S->num_tuples;

  // uint32_t N = ceil(numS * 0.08);
  // uint32_t N = findMaxBins(numR, 0.001, 1e-6);
  // PREV_POW_2(N);
  // printf("(DISTRIBUTE)   numR=%lu,bins=%u\n", numR, N);

  // const uint32_t MASK = (N - 1) << (NUM_RADIX_BITS);
  const uint32_t MASK = (BINS - 1) << (NUM_RADIX_BITS);


  next = (int *)malloc(sizeof(int) * numR);
  // bucket = (int *)calloc(N, sizeof(int));
  bucket = (int *)calloc(BINS, sizeof(int));


  struct row_t *Rtuples = R->tuples;
  for (uint32_t i = 0; i < numR;) {
    uint32_t idx = HASH_BIT_MODULO(R->tuples[i].hashKey, MASK, NUM_RADIX_BITS);
    next[i] = bucket[idx];
    bucket[idx] = ++i; /* we start pos's from 1 instead of 0 */
  }

  struct row_t *Stuples = S->tuples;
  for (uint32_t i = 0; i < numS; i++) {
    uint32_t idx = HASH_BIT_MODULO(Stuples[i].hashKey, MASK, NUM_RADIX_BITS);
    uint32_t outPos;
    int match;
    for (int hit = bucket[idx]; hit > 0; hit = next[hit - 1]) {
      outPos = isIdxS ? Stuples[i].idx
                      : Rtuples[hit - 1].idx; // branching on public knowledge
      if (isIdxS) {
        match = (Rtuples[hit - 1].idx == outPos) &
                (Rtuples[hit - 1].cntExpand != 0);
        __m256i vSrc = _mm256_loadu_si256((const __m256i *)(&Rtuples[hit - 1]));
        __m256i vDst = _mm256_loadu_si256(
            (const __m256i *)(&expanded->tuples[Stuples[i].idx]));
        __m256i m = _mm256_set1_epi64x(-(uint64_t)match);
        __m256i res = _mm256_or_si256(_mm256_and_si256(vSrc, m),
                                      _mm256_andnot_si256(m, vDst));
        _mm256_storeu_si256((__m256i *)(&expanded->tuples[Stuples[i].idx]),
                            res);
      } else {
        match = (Stuples[i].idx == outPos) & (Stuples[i].cntExpand != 0);
        __m256i vSrc = _mm256_loadu_si256((const __m256i *)(&Stuples[i]));
        __m256i vDst = _mm256_loadu_si256(
            (const __m256i *)(&expanded->tuples[Rtuples[hit - 1].idx]));
        __m256i m = _mm256_set1_epi64x(-(uint64_t)match); // 256-bit broadcast
        __m256i res = _mm256_or_si256(_mm256_and_si256(vSrc, m),
                                      _mm256_andnot_si256(m, vDst));
        _mm256_storeu_si256(
            (__m256i *)(&expanded->tuples[Rtuples[hit - 1].idx]), res);
      }
    }
  }

  /* clean up temp */
  free(bucket);
  free(next);

  return 0;
}

/**
 * Radix clustering algorithm (originally described by Manegold et al)
 * The algorithm mimics the 2-pass radix clustering algorithm from
 * Kim et al. The difference is that it does not compute
 * prefix-sum, instead the sum (offset in the code) is computed iteratively.
 *
 * @warning This method puts padding between clusters, see
 * radix_cluster_nopadding for the one without padding.
 *
 * @param outRel [out] result of the partitioning
 * @param inRel [in] input relation
 * @param hist [out] number of tuples in each partition
 * @param R cluster bits
 * @param D radix bits per pass
 * @returns tuples per partition.
 */
static void radix_cluster(struct table_t *outRel, struct table_t *inRel,
                          int64_t *hist, int R, int D) {
  uint64_t i;
  uint32_t M = ((1 << D) - 1) << R;
  uint32_t offset;
  uint32_t fanOut = 1 << D;

  /* the following are fixed size when D is same for all the passes,
     and can be re-used from call to call. Allocating in this function
     just in case D differs from call to call. */
  uint32_t dst[fanOut];

  /* count tuples per cluster */
  for (i = 0; i < inRel->num_tuples; i++) {
    uint64_t idx = HASH_BIT_MODULO(inRel->tuples[i].hashKey, M, R);
    // uint64_t idx = partition_idx(inRel->tuples[i].key, M, R);
    hist[idx]++;
  }
  offset = 0;
  /* determine the start and end of each cluster depending on the counts. */
  for (i = 0; i < fanOut; i++) {
    /* dst[i]      = outRel->tuples + offset; */
    /* determine the beginning of each partitioning by adding some
       padding to avoid L1 conflict misses during scatter. */
    dst[i] = (uint32_t)(offset + i * SMALL_PADDING_TUPLES);
    offset += hist[i];
  }

  /* copy tuples to their corresponding clusters at appropriate offsets */
  for (i = 0; i < inRel->num_tuples; i++) {
    uint32_t idx = HASH_BIT_MODULO(inRel->tuples[i].hashKey, M, R);
    // uint32_t idx = partition_idx(inRel->tuples[i].key, M, R);
    outRel->tuples[dst[idx]] = inRel->tuples[i];
    ++dst[idx];
  }
}

/**
 * This function implements the radix clustering of a given input
 * relations. The relations to be clustered are defined in task_t and after
 * clustering, each partition pair is added to the join_queue to be joined.
 *
 * @param task description of the relation to be partitioned
 * @param join_queue task queue to add join tasks after clustering
 */
static void serial_radix_partition(task_t *const task, task_queue_t *join_queue,
                                   const int R, const int D) {
  int i;
  uint64_t offsetR = 0, offsetS = 0;
  const int fanOut = 1 << D; /*(NUM_RADIX_BITS / NUM_PASSES);*/
  int64_t *outputR, *outputS;

  outputR = (int64_t *)calloc(fanOut + 1, sizeof(int64_t));
  outputS = (int64_t *)calloc(fanOut + 1, sizeof(int64_t));
  /* TODO: measure the effect of memset() */
  /* memset(outputR, 0, fanOut * sizeof(int32_t)); */
  radix_cluster(&task->tmpR, &task->relR, outputR, R, D);

  /* memset(outputS, 0, fanOut * sizeof(int32_t)); */
  radix_cluster(&task->tmpS, &task->relS, outputS, R, D);

  /* task_t t; */
  for (i = 0; i < fanOut; i++) {
    if (outputR[i] > 0 && outputS[i] > 0) {
      task_t *t = task_queue_get_slot_atomic(join_queue);
      t->relR.num_tuples = outputR[i];
      t->relR.tuples = task->tmpR.tuples + offsetR + i * SMALL_PADDING_TUPLES;
      t->tmpR.tuples = task->relR.tuples + offsetR + i * SMALL_PADDING_TUPLES;
      offsetR += outputR[i];

      t->relS.num_tuples = outputS[i];
      t->relS.tuples = task->tmpS.tuples + offsetS + i * SMALL_PADDING_TUPLES;
      t->tmpS.tuples = task->relS.tuples + offsetS + i * SMALL_PADDING_TUPLES;
      offsetS += outputS[i];

      /* task_queue_copy_atomic(join_queue, &t); */
      task_queue_add_atomic(join_queue, t);
    } else {
      offsetR += outputR[i];
      offsetS += outputS[i];
    }
  }
  free(outputR);
  free(outputS);
}

/**
 * This function implements the parallel radix partitioning of a given input
 * relation. Parallel partitioning is done by histogram-based relation
 * re-ordering as described by Kim et al. Parallel partitioning method is
 * commonly used by all parallel radix join algorithms.
 *
 * @param part description of the relation to be partitioned
 */
static void parallel_radix_partition(part_t *const part) {
  const struct row_t *rel = part->rel;
  int32_t **hist = part->hist;
  int32_t *output = part->output;

  const uint32_t my_tid = part->thrargs->my_tid;
  const uint32_t nthreads = part->thrargs->nthreads;
  const uint32_t size = part->num_tuples;

  const int32_t R = part->R;
  const int32_t D = part->D;
  const uint32_t fanOut = 1 << D;
  const uint32_t MASK = (fanOut - 1) << R;
  const uint32_t padding = part->padding;

  // if(my_tid == 0)
  // {
  //     printf("Radix partitioning. R=%d, D=%d, fanout=%d, MASK=%d\n",
  //            R, D, fanOut, MASK);
  // }

  int32_t sum = 0;
  uint32_t i, j;
  int rv = 0;

  int32_t dst[fanOut + 1];

  /* compute local histogram for the assigned region of rel */
  /* compute histogram */
  int32_t *my_hist = hist[my_tid];

  for (i = 0; i < size; i++) {
    uint32_t idx = HASH_BIT_MODULO(rel[i].hashKey, MASK, R);
    // uint32_t idx = partition_idx(rel[i].key, MASK, R);
    my_hist[idx]++;
  }

  /* compute local prefix sum on hist */
  for (i = 0; i < fanOut; i++) {
    sum += my_hist[i];
    my_hist[i] = sum;
  }

  /* wait at a barrier until each thread complete histograms */
  barrier_arrive(part->thrargs->barrier, rv);

  /* determine the start and end of each cluster */
  for (i = 0; i < my_tid; i++) {
    for (j = 0; j < fanOut; j++)
      output[j] += hist[i][j];
  }
  for (i = my_tid; i < nthreads; i++) {
    for (j = 1; j < fanOut; j++)
      output[j] += hist[i][j - 1];
  }

  for (i = 0; i < fanOut; i++) {
    output[i] += i * padding; // PADDING_TUPLES;
    dst[i] = output[i];
  }
  output[fanOut] = part->total_tuples + fanOut * padding; // PADDING_TUPLES;

  struct row_t *tmp = part->tmp;

  /* Copy tuples to their corresponding clusters */
  for (i = 0; i < size; i++) {
    uint32_t idx = HASH_BIT_MODULO(rel[i].hashKey, MASK, R);
    tmp[dst[idx]] = rel[i];
    ++dst[idx];
  }
}

/**
 * The main thread of parallel radix join. It does partitioning in parallel with
 * other threads and during the join phase, picks up join tasks from the task
 * queue and calls appropriate JoinFunction to compute the join task.
 *
 * @param param
 *
 * @return
 */
static void *prj_thread(void *param) {
  arg_t_radix *args = (arg_t_radix *)param;
  int32_t my_tid = args->my_tid;

  const int fanOut = 1 << (NUM_RADIX_BITS / NUM_PASSES);
  const int R = (NUM_RADIX_BITS / NUM_PASSES);
  const int D = (NUM_RADIX_BITS - (NUM_RADIX_BITS / NUM_PASSES));
  const int thresh1 = (const int)(MAX((1 << D), (1 << R)) *
                                  THRESHOLD1((unsigned long)args->nthreads));

  // if (args->my_tid == 0)
  // {
  //     printf("NUM_PASSES=%d, RADIX_BITS=%d\n", NUM_PASSES, NUM_RADIX_BITS);
  //     printf("fanOut = %d, R = %d, D = %d, thresh1 = %d\n", fanOut, R, D,
  //     thresh1);
  // }
  uint64_t results = 0;
  int i;
  int rv = 0;

  part_t part;
  task_t *task;
  task_queue_t *part_queue;
  task_queue_t *join_queue;

  int32_t *outputR = (int32_t *)calloc((fanOut + 1), sizeof(int32_t));
  int32_t *outputS = (int32_t *)calloc((fanOut + 1), sizeof(int32_t));
  malloc_check((void *)(outputR && outputS));

  part_queue = args->part_queue;
  join_queue = args->join_queue;

  args->histR[my_tid] = (int32_t *)calloc(fanOut, sizeof(int32_t));
  args->histS[my_tid] = (int32_t *)calloc(fanOut, sizeof(int32_t));

  /* in the first pass, partitioning is done together by all threads */

  args->parts_processed = 0;

  /* wait at a barrier until each thread starts and then start the timer */
  barrier_arrive(args->barrier, rv);

  /********** 1st pass of multi-pass partitioning ************/
  part.R = 0;
  part.D = NUM_RADIX_BITS / NUM_PASSES;
  part.thrargs = args;
  part.padding = PADDING_TUPLES;

  /* 1. partitioning for relation R */
  part.rel = args->relR;
  part.tmp = args->tmpR;
  part.hist = args->histR;
  part.output = outputR;
  part.num_tuples = args->numR;
  part.total_tuples = args->totalR;
  part.relidx = 0;

  parallel_radix_partition(&part);

  /* 2. partitioning for relation S */
  part.rel = args->relS;
  part.tmp = args->tmpS;
  part.hist = args->histS;
  part.output = outputS;
  part.num_tuples = args->numS;
  part.total_tuples = args->totalS;
  part.relidx = 1;

  parallel_radix_partition(&part);

  /* wait at a barrier until each thread copies out */
  barrier_arrive(args->barrier, rv);

  /********** end of 1st partitioning phase ******************/

  /* 3. first thread creates partitioning tasks for 2nd pass */
  if (my_tid == 0) {
    for (i = 0; i < fanOut; i++) {
      int32_t ntupR = outputR[i + 1] - outputR[i] - (int32_t)PADDING_TUPLES;
      int32_t ntupS = outputS[i + 1] - outputS[i] - (int32_t)PADDING_TUPLES;

      if (ntupR > 0 && ntupS > 0) {
        task_t *t = task_queue_get_slot(part_queue);

        t->relR.num_tuples = t->tmpR.num_tuples = ntupR;
        t->relR.tuples = args->tmpR + outputR[i];
        t->tmpR.tuples = args->tmpR2 + outputR[i];

        t->relS.num_tuples = t->tmpS.num_tuples = ntupS;
        t->relS.tuples = args->tmpS + outputS[i];
        t->tmpS.tuples = args->tmpS2 + outputS[i];

        task_queue_add(part_queue, t);
      }
    }

    /* debug partitioning task queue */
    // printf("Pass-2: # partitioning tasks = %d\n", part_queue->count);
  }

  /* wait at a barrier until first thread adds all partitioning tasks */
  barrier_arrive(args->barrier, rv);

  /************ 2nd pass of multi-pass partitioning ********************/
  /* 4. now each thread further partitions and add to join task queue **/

#if NUM_PASSES == 1
  /* If the partitioning is single pass we directly add tasks from pass-1 */
  task_queue_t *swap = join_queue;
  join_queue = part_queue;
  /* part_queue is used as a temporary queue for handling skewed parts */
  part_queue = swap;

#elif NUM_PASSES == 2

  while ((task = task_queue_get_atomic(part_queue))) {
    serial_radix_partition(task, join_queue, R, D);
  }

#else
#warning Only 2-pass partitioning is implemented, set NUM_PASSES to 2!
#endif

  free(outputR);
  free(outputS);

  /* wait at a barrier until all threads add all join tasks */
  barrier_arrive(args->barrier, rv);

  // if(my_tid == 0)
  // {
  //     printf("Number of join tasks = %d\n", join_queue->count);
  // }

  output_list_t *output;

  while ((task = task_queue_get_atomic(join_queue))) {
    // /* do the actual join. join method differs for different algorithms,
    //    i.e. bucket chaining, histogram-based, histogram-based with simd &
    //    prefetching  */
    results += args->join_function(&task->relR, &task->relS, &task->tmpR,
                                   &output, args->expanded_tbl, args->isIdxS);
    args->parts_processed++;
  }

  args->result = results;
  barrier_arrive(args->barrier, rv);
  return 0;
}

/**
 * The template function for different joins: Basically each parallel radix join
 * has a initialization step, partitioning step and build-probe steps. All our
 * parallel radix implementations have exactly the same initialization and
 * partitioning steps. Difference is only in the build-probe step. Here are all
 * the parallel radix join implemetations and their Join (build-probe)
 * functions:
 *
 * - PRO,  Parallel Radix Join Optimized --> bucket_chaining_join()
 * - PRH,  Parallel Radix Join Histogram-based --> histogram_join()
 * - PRHO, Parallel Radix Histogram-based Optimized ->
 * histogram_optimized_join()
 */
static result_t *join_init_run(struct table_t *relR, struct table_t *relS,
                               JoinFunctionIdx jf, int nthreads,
                               struct table_t *expanded, bool isIdxS) {
  int i, rv;
  pthread_t tid[nthreads];
  pthread_barrier_t barrier;

  arg_t_radix args[nthreads];

  int32_t **histR, **histS;
  struct row_t *tmpRelR, *tmpRelS;
  struct row_t *tmpRelR2, *tmpRelS2;
  uint64_t numperthr[2];
  int64_t result = 0;

  task_queue_t *part_queue, *join_queue;

  part_queue = task_queue_init(FANOUT_PASS1);
  join_queue = task_queue_init((1 << NUM_RADIX_BITS));

  /* allocate temporary space for partitioning */
  tmpRelR = (struct row_t *)alloc_aligned(
      relR->num_tuples * sizeof(struct row_t) + RELATION_PADDING);
  tmpRelS = (struct row_t *)alloc_aligned(
      relS->num_tuples * sizeof(struct row_t) + RELATION_PADDING);
  malloc_check((void *)(tmpRelR && tmpRelS));

  tmpRelR2 = (struct row_t *)alloc_aligned(
      relR->num_tuples * sizeof(struct row_t) + RELATION_PADDING);
  tmpRelS2 = (struct row_t *)alloc_aligned(
      relS->num_tuples * sizeof(struct row_t) + RELATION_PADDING);

  /* allocate histograms arrays, actual allocation is local to threads */
  histR = (int32_t **)alloc_aligned(nthreads * sizeof(int32_t *));
  histS = (int32_t **)alloc_aligned(nthreads * sizeof(int32_t *));
  malloc_check((void *)(histR && histS));

  rv = pthread_barrier_init(&barrier, NULL, nthreads);
  if (rv != 0) {
    printf("Couldn't create the barrier\n");
    exit(EXIT_FAILURE);
  }

  /* first assign chunks of relR & relS for each thread */
  numperthr[0] = relR->num_tuples / nthreads;
  numperthr[1] = relS->num_tuples / nthreads;

  result_t *joinresult;
  joinresult = (result_t *)malloc(sizeof(result_t));
  joinresult->resultlist =
      (threadresult_t *)malloc(sizeof(threadresult_t) * nthreads);
  for (i = 0; i < nthreads; i++) {
    args[i].relR = relR->tuples + i * numperthr[0];
    args[i].tmpR = tmpRelR;
    args[i].histR = histR;

    args[i].relS = relS->tuples + i * numperthr[1];
    args[i].tmpS = tmpRelS;
    args[i].histS = histS;

    /* Store original array pointers for propagation */
    args[i].origRelR = relR->tuples;
    args[i].origRelS = relS->tuples;
    args[i].tmpR2 = tmpRelR2;
    args[i].tmpS2 = tmpRelS2;

    args[i].expanded_tbl = expanded;
    args[i].isIdxS = isIdxS;

    args[i].numR = (i == (nthreads - 1)) ? (relR->num_tuples - i * numperthr[0])
                                         : numperthr[0];
    args[i].numS = (i == (nthreads - 1)) ? (relS->num_tuples - i * numperthr[1])
                                         : numperthr[1];
    args[i].totalR = relR->num_tuples;
    args[i].totalS = relS->num_tuples;

    args[i].my_tid = i;
    args[i].part_queue = part_queue;
    args[i].join_queue = join_queue;

    args[i].barrier = &barrier;
    args[i].join_function = jf;
    args[i].nthreads = nthreads;

    rv = pthread_create(&tid[i], NULL, prj_thread, (void *)&args[i]);

    if (rv) {
      printf("return code from pthread_create() is %d\n", rv);
      exit(EXIT_FAILURE);
    }
  }

  /* wait for threads to finish */
  for (i = 0; i < nthreads; i++) {
    pthread_join(tid[i], NULL);
    result += args[i].result;
  }

  joinresult->totalresults = result;
  joinresult->nthreads = nthreads;

  /* clean up */
  for (i = 0; i < nthreads; i++) {
    free(histR[i]);
    free(histS[i]);
  }
  free(histR);
  free(histS);

  /* Clean up task queues more carefully */
  if (part_queue) {
    task_queue_free(part_queue);
  }
  if (join_queue) {
    task_queue_free(join_queue);
  }

  if (tmpRelR) {
    free(tmpRelR);
  }
  if (tmpRelS) {
    free(tmpRelS);
  }

  if (tmpRelR2) {
    free(tmpRelR2);
  }
  if (tmpRelS2) {
    free(tmpRelS2);
  }

  return joinresult;
}

result_t *RHO_idx(struct table_t *relR, struct table_t *relS, int nthreads,
                  struct table_t *expanded, bool isIdxS) {
  return join_init_run(relR, relS, bucket_chaining_join_idx, nthreads, expanded,
                       isIdxS);
}
