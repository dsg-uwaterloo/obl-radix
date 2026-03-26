#pragma once

#include <cstdlib>

#include "external_memory/dynamicvector.hpp"
#include "external_memory/noncachedvector.hpp"
#include "external_memory/par_io.hpp"
#include "param_select.hpp"
#include "sort_building_blocks.hpp"

// Parallel flex-way butterfly o-shuffle (random permutation).
//
// This is adapted from the `oblsort` implementation and keeps the iterator-based
// API used by NFK-OnOff (e.g., RawVector iterators). It pins OpenMP to a caller
// specified thread count via KWayButterflyOShuffleFixedThreads().

namespace EM::Algorithm {

template <typename IOIterator>
class ButterflyOShuffleSorter {
private:
  using T = typename std::iterator_traits<IOIterator>::value_type;
  using WrappedT = TaggedT<T>;
  using IOVector =
      typename std::remove_reference<decltype(*(IOIterator::getNullVector()))>::type;

  uint64_t Z = 0;
  uint64_t numTotalBucket = 0;
  uint64_t numRealPerBucket = 0;

  uint64_t numBucketFit = 0;
  uint64_t numElementFit = 0;

  KWayButterflyParams KWayParams{};

  RWManager<typename IOVector::PrefetchReader, typename IOVector::Iterator>
      inputReaderManager;
  RWManager<typename IOVector::Writer, typename IOVector::Iterator>
      outputWriterManager;

  ThreadSafeStack<uint64_t> freeTempIndices;
  WrappedT *batch = nullptr;
  uint8_t *tempBegin = nullptr;
  uint64_t tempSize = 0;

  void KWayButterflySortBasicNonRecursive(WrappedT *begin, WrappedT *end,
                                         size_t ioLayer,
                                         size_t innerLayer) {
    const uint64_t numElement = (uint64_t)(end - begin);
    const uint64_t numBucket = numElement / Z;

    for (uint64_t layer = 0, stride = 1; layer <= innerLayer; ++layer) {
      const uint64_t way = KWayParams.ways[ioLayer][layer];
      const uint64_t wayBucket = numBucket / way;

#pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < wayBucket; ++i) {
        thread_local RandGen local_rng;
        uint64_t tempIdx;
        if (!freeTempIndices.Pop(tempIdx)) {
          printf("freeTempIndices is empty\n");
          abort();
        }

        if (layer == 0 && ioLayer == 0) {
          // First layer tags input and pads with dummies.
          auto *inputReader = inputReaderManager.getRW();
          bool overFlag = (inputReader == nullptr);

          for (uint64_t j = 0; j < way; ++j) {
            auto it = begin + (i * way + j) * Z;
            if (overFlag) {
              for (; j < way; ++j) {
                for (uint64_t off = 0; off < Z; ++off, ++it)
                  it->setDummy();
              }
              break;
            }

            uint64_t off = 0;
            for (; off < numRealPerBucket; ++off, ++it) {
              if (inputReader->eof()) {
                inputReaderManager.returnRW(inputReader);
                inputReader = inputReaderManager.getRW();
                if (inputReader == nullptr) {
                  overFlag = true;
                  break;
                }
              }
              it->setData(inputReader->read(), local_rng);
            }
            for (; off < Z; ++off, ++it)
              it->setDummy();
          }

          if (inputReader)
            inputReaderManager.returnRW(inputReader);
        }

        const uint64_t groupIdx = i / stride;
        const uint64_t groupOffset = i % stride;

        WrappedT *KWayIts[8];
        for (uint64_t j = 0; j < way; ++j) {
          KWayIts[j] =
              begin + ((j + groupIdx * way) * stride + groupOffset) * Z;
        }

        const size_t tempBucketsSize = (size_t)way * (size_t)Z * sizeof(WrappedT);
        uint8_t *temp = tempBegin + tempIdx * tempSize;
        uint8_t *marks = temp + tempBucketsSize;
        MergeSplitKWay(KWayIts, way, Z, (WrappedT *)temp, marks);

        freeTempIndices.Push(tempIdx);
      }

      stride *= way;
    }
  }

public:
  ButterflyOShuffleSorter(IOIterator inputBeginIt, IOIterator inputEndIt,
                          uint32_t inAuth, uint64_t heapSize, int fixedThreads)
      : numElementFit(heapSize / sizeof(WrappedT)) {
    const size_t size = (size_t)(inputEndIt - inputBeginIt);

    if (fixedThreads < 1)
      fixedThreads = 1;
    omp_set_nested(0);
    omp_set_num_threads(fixedThreads);
    thread_count = fixedThreads;

    // WrappedT (TaggedT<T>) is over-aligned (typically 32B). Plain malloc only
    // guarantees max_align_t (often 16B) which can lead to crashes when the
    // algorithm uses aligned SIMD loads/stores.
    void *raw = nullptr;
    const size_t alignment = alignof(WrappedT);
    const int allocRc = posix_memalign(&raw, alignment, (size_t)heapSize);
    batch = (WrappedT *)raw;
    if (allocRc != 0 || !batch) {
      printf("batch allocation failed\n");
      abort();
    }

    // Pick parameters while keeping parallelism fixed/public.
    KWayParams = bestKWayButterflyParamsFixedThreads(
        size, numElementFit, (int64_t)sizeof(T), -60, fixedThreads);

    // RawVector iterators have no page size; disable alignment.
    inputReaderManager.template init<false>(inputBeginIt, inputEndIt, inAuth,
                                            fixedThreads);
    Z = KWayParams.Z;

    // Reserve per-thread scratch: (8 buckets of temp + marks) per worker.
    const uint64_t tempElementFit =
        divRoundUp(Z * 8 * (sizeof(WrappedT) + 2), sizeof(WrappedT));
    if (numElementFit <= tempElementFit * (uint64_t)fixedThreads) {
      printf("Heap too small for fixedThreads scratch\n");
      abort();
    }
    numElementFit -= tempElementFit * (uint64_t)fixedThreads;

    freeTempIndices.init((uint64_t)fixedThreads);
    for (uint64_t i = 0; i < (uint64_t)fixedThreads; ++i) {
      freeTempIndices.Push(i);
    }

    tempBegin = (uint8_t *)batch + numElementFit * sizeof(WrappedT);
    tempSize = tempElementFit * sizeof(WrappedT);

    numBucketFit = numElementFit / Z;
    numTotalBucket = KWayParams.totalBucket;
    numRealPerBucket = 1 + (size - 1) / numTotalBucket;

    outputWriterManager.template init<false>(inputBeginIt, inputEndIt, inAuth + 1,
                                             fixedThreads);
  }

  ~ButterflyOShuffleSorter() {
    if (batch) {
      free(batch);
      batch = nullptr;
    }
  }

  void sort(IOIterator begin, IOIterator end) {
    (void)begin;
    (void)end;

    // This implementation uses the same batch-buffered butterfly network as
    // `oblsort` and writes out the permutation to the output writer managers.
    EM::DynamicPageVector::Vector<TaggedT<T>> v(numTotalBucket * Z, Z);

    auto vBegin = v.begin();
    auto vEnd = v.end();

    // Run the butterfly network layer by layer, moving data through v.
    for (size_t ioLayer = 0; ioLayer < KWayParams.ways.size(); ++ioLayer) {
      const bool isLastLayer = (ioLayer + 1 == KWayParams.ways.size());
      const size_t numInternalWay = getVecProduct(KWayParams.ways[ioLayer]);

      size_t fetchInterval = 1;
      for (size_t layer = 0; layer < ioLayer; ++layer) {
        fetchInterval *= getVecProduct(KWayParams.ways[layer]);
      }

      const size_t totalSize = (size_t)(vEnd - vBegin);
      const size_t numInterval = totalSize / Z / fetchInterval;

      const size_t bucketPerBatch =
          std::min(totalSize / Z, numBucketFit / numInternalWay * numInternalWay);
      if (bucketPerBatch == 0) {
        printf("bucketPerBatch == 0 (heap too small)\n");
        abort();
      }

      const size_t batchSize = bucketPerBatch * Z;
      const size_t batchCount = divRoundUp(totalSize, batchSize);

      for (size_t batchIdx = 0; batchIdx < batchCount; ++batchIdx) {
        size_t bucketThisBatch = bucketPerBatch;
        if (batchIdx + 1 == batchCount) {
          bucketThisBatch = totalSize / Z - bucketPerBatch * (batchCount - 1);
        }

        if (ioLayer > 0) {
#pragma omp parallel for schedule(static)
          for (size_t bucketIdx = 0; bucketIdx < bucketThisBatch; ++bucketIdx) {
            const size_t bucketGlobalIdx = batchIdx * bucketPerBatch + bucketIdx;
            auto extBeginIt =
                vBegin +
                (bucketGlobalIdx / numInterval +
                 (bucketGlobalIdx % numInterval) * fetchInterval) *
                    Z;
            auto intBeginIt = batch + bucketIdx * Z;
            CopyIn(extBeginIt, extBeginIt + Z, intBeginIt, (uint32_t)ioLayer - 1);
          }
        }

        KWayButterflySortBasicNonRecursive(batch, batch + bucketThisBatch * Z,
                                           ioLayer,
                                           KWayParams.ways[ioLayer].size() - 1);

        if (isLastLayer) {
          const auto cmpTag = [](const auto &a, const auto &b) {
            return a.tag < b.tag;
          };

#pragma omp parallel for schedule(static)
          for (size_t i = 0; i < bucketThisBatch; ++i) {
            auto it = batch + i * Z;
            BitonicSort(it, it + Z, cmpTag);

            auto *outputWriter = outputWriterManager.getRW();
            if (!outputWriter) {
              printf("outputWriter is null\n");
              abort();
            }
            for (auto fromIt = it; fromIt != it + Z; ++fromIt) {
              if (!fromIt->isDummy()) {
                if (outputWriter->eof()) {
                  outputWriterManager.returnRW(outputWriter);
                  outputWriter = outputWriterManager.getRW();
                  if (!outputWriter) {
                    printf("outputWriter is null\n");
                    abort();
                  }
                }
                outputWriter->write(fromIt->getData());
              }
            }
            outputWriterManager.returnRW(outputWriter);
          }
        } else {
#pragma omp parallel for schedule(static)
          for (size_t bucketIdx = 0; bucketIdx < bucketThisBatch; ++bucketIdx) {
            const size_t bucketGlobalIdx = batchIdx * bucketPerBatch + bucketIdx;
            auto extBeginIt =
                vBegin +
                (bucketGlobalIdx / numInterval +
                 (bucketGlobalIdx % numInterval) * fetchInterval) *
                    Z;
            auto intBeginIt = batch + bucketIdx * Z;
            CopyOut(intBeginIt, intBeginIt + Z, extBeginIt, (uint32_t)ioLayer);
          }
        }
      }
    }

    outputWriterManager.flush();
  }
};

template <class Iterator>
void KWayButterflyOShuffleFixedThreads(Iterator begin, Iterator end,
                                       uint32_t inAuth, uint64_t heapSize,
                                       int numThreads) {
  if constexpr (Iterator::random_access) {
    const size_t N = (size_t)(end - begin);
    if (N <= 512) {
      OrShuffle(begin, end);
      return;
    }
  }

  ButterflyOShuffleSorter<Iterator> sorter(begin, end, inAuth, heapSize,
                                           numThreads);
  sorter.sort(begin, end);
}

template <class Iterator>
void KWayButterflyOShuffle(Iterator begin, Iterator end, uint32_t inAuth,
                           uint64_t heapSize) {
  // Default behavior: keep using the OpenMP thread_count decided by the solver.
  // NFK-OnOff uses the fixed-thread entrypoint instead.
  KWayButterflyOShuffleFixedThreads(begin, end, inAuth, heapSize, thread_count);
}

template <typename Vec>
void KWayButterflyOShuffle(Vec &vec, uint32_t inAuth, uint64_t heapSize) {
  KWayButterflyOShuffle(vec.begin(), vec.end(), inAuth, heapSize);
}

} // namespace EM::Algorithm
