#pragma once
#include "apps_common.hpp"

namespace Apps {
    template <typename K, typename V, typename PositionType = uint64_t>
struct ParOMap {
 private:
//   using BaseMap = OHashMap<K, V, true, PositionType, true>;

  struct KVPair {
    K key;
    V value;
  };


  // underlying shards
//   std::vector<BaseMap> shards;
  // the size of each shard
  PositionType shardSize = 0;
  // random salt for hashing keys to shards
  uint8_t randSalt[16];

  // the size of hash range for each shard
  uint64_t shardHashRange;

  // the capacity of the map
  PositionType _size = 0;

  // whether the map is initialized
  bool inited = false;

  uint32_t numShards = 1;

  /**
   * @brief translate a hash to a shard index
   *
   * @param hash the hash value
   * @return uint32_t the shard index
   */
  INLINE uint32_t getShardByHash(uint64_t hash) {
    return (uint32_t)(hash / shardHashRange);
  }

  /**
   * @brief Calculate an upper bound of the number of queries each shard can
   * take, given an allowed failure probability.
   *
   * @param batchSize The number of queries in the batch
   * @param shardCount The number of shards
   * @param logFailProb Logarithm of the allowed failure probability
   * @return uint64_t Upper bound of the number of queries each shard can
   * take
   */
  static uint64_t maxQueryPerShard(uint64_t batchSize, uint64_t shardCount,
                                   double logFailProb = -40) {
    auto satisfy = [&](uint64_t n) {
      double logSf =
          EM::Algorithm::binomLogSf(n, batchSize, 1.0 / (double)shardCount);
      return logSf < logFailProb;
    };
    return EM::Algorithm::lowerBound(divRoundUp(batchSize, shardCount), batchSize,
                                 satisfy);
  }

  /**
   * @brief Calculate the maximum number of real elements each bkt can hold
   * initially when routing data through the butterfly network during the
   * initialization of the oblivious maps.
   *
   * @param bktSize The size of each bkt
   * @param shardCount The number of shards
   * @param logFailProb Logarithm of the allowed failure probability
   * @return uint64_t The maximum number of real elements each bkt may hold
   */
  static uint64_t numRealPerBkt(uint64_t bktSize, uint64_t shardCount,
                                double logFailProb = -60) {
    auto satisfy = [&](uint64_t numDummy) {
      double logSf = EM::Algorithm::binomLogSf(
          bktSize, (bktSize - numDummy) * shardCount, 1.0 / shardCount);
      return logSf < logFailProb;
    };
    return bktSize - EM::Algorithm::lowerBound(1UL, bktSize - 1, satisfy);
  }

  /**
   * @brief Factorize the number of shards into a list of factors between 2
   * and 8. We want to have as few factors as possible and the factors should be
   * close to each other. The method will throw an error if the number of shards
   * doesn't meet the requirements.
   *
   * @param shardCount The number of shards
   * @return std::vector<uint64_t> The list of factors
   */
  static std::vector<uint64_t> factorizeShardCount(uint64_t shardCount) {
    if (shardCount == 1) {
      throw std::runtime_error(
          "shardCount should be at least 2 for init with "
          "reader");
    }
    if (shardCount > 64) {
      throw std::runtime_error(
          "shardCount should be no more than 64 for init with "
          "reader");
    }
    switch (shardCount) {
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
      case 8:
        return {shardCount};
      case 9:
        return {3, 3};
      case 10:
        return {2, 5};
      case 12:
        return {3, 4};
      case 14:
        return {2, 7};
      case 15:
        return {3, 5};
      case 16:
        return {4, 4};
      case 18:
        return {3, 6};
      case 20:
        return {4, 5};
      case 21:
        return {3, 7};
      case 24:
        return {3, 8};
      case 25:
        return {5, 5};
      case 27:
        return {3, 3, 3};
      case 28:
        return {4, 7};
      case 30:
        return {5, 6};
      case 32:
        return {4, 8};
      case 35:
        return {5, 7};
      case 36:
        return {6, 6};
      case 40:
        return {5, 8};
      case 42:
        return {6, 7};
      case 45:
        return {3, 3, 5};
      case 48:
        return {6, 8};
      case 49:
        return {7, 7};
      case 50:
        return {2, 5, 5};
      case 54:
        return {3, 3, 6};
      case 56:
        return {7, 8};
      case 60:
        return {3, 4, 5};
      case 63:
        return {3, 3, 7};
      case 64:
        return {8, 8};
      default:
        throw std::runtime_error(
            "shardCount should be a power of 2 between 2 and 64 for init with "
            "reader");
    }
  }

 public:
  ParOMap() {}

  /**
   * @brief Construct a new parallel oblivious map object, does not allocate
   * resources. Must call Init or InitFromReader before using the map.
   *
   * @param mapSize The capacity of the map
   * @param shardCount The number of shards, should be between 2 and 64 and be a
   * factor of numbers between 2 and 8 to support Initialization. Please use
   * GetSuitableShardCount method to get a suitable shard count.
   */
  ParOMap(PositionType mapSize, uint64_t shardCount) {
    SetSize(mapSize, shardCount);
  }

  /**
   * @brief Set the size of the map and the number of shards, does not allocate
   * resources.
   *
   * @param mapSize The capacity of the map
   * @param shardCount The number of shards, should be a power of 2 to support
   * InitFromReader. Suggested to be the number of available threads / 2.
   */
  void SetSize(PositionType mapSize, uint64_t shardCount) {
    if (shardCount == 0) {
      throw std::runtime_error("shardCount should be positive");
    }
    if (shardCount > 64) {
      throw std::runtime_error("shardCount should be no more than 64");
    }
    _size = mapSize;
    // shards.resize(shardCount);
    numShards = shardCount;
    shardSize = (PositionType)maxQueryPerShard(mapSize, shardCount, -60);
    memset(randSalt, 0, 16); // set as 0 for testing
    shardHashRange = (UINT64_MAX - 1) / shardCount + 1;
  }

  /**
   * @brief Get a suitable shard count given the number of threads.
   *
   * @param numThreads The number of threads available for parallelization.
   * @param emptyInit Whether the omap will be initialized with empty data.
   * @return uint32_t The suitable shard count.
   */
  static uint32_t GetSuitableShardCount(uint32_t numThreads,
                                        bool emptyInit = false) {
    if (numThreads <= 4) {
      // too few threads, use 2 shards
      return 2;
    }
    uint32_t suitableShardCount = numThreads / 2;
    if (suitableShardCount > 64) {
      suitableShardCount = 64;
    }
    if (emptyInit) {
      // don't need to factorize
      return suitableShardCount;
    }
    for (; suitableShardCount >= 1; --suitableShardCount) {
      try {
        factorizeShardCount(suitableShardCount);
        // if no exception is thrown, then the shard count is valid
        return suitableShardCount;
      } catch (const std::runtime_error& e) {
        continue;
      }
    }
    return 2;
  }

  /**
   * @brief Return the capacity of the map.
   *
   * @return PositionType
   */
  PositionType size() const { return _size; }


  /**
   * @brief An object that stores the curret state of initialization.
   * Faciliates initialization in a streaming fashion. We route the data in
   * parallel through a multi-way butterfly network to load balance them to each
   * shard.
   * Might throw error with some negligible probability (when some bucket
   * overflows), in which case one may retry.
   *
   */
  struct InitContext {
    using Element = EM::Algorithm::TaggedT<KVPair>;
    // using NonObliviousOHashMap = OHashMap<K, V, false, PositionType>;
    ParOMap& omap;                                // reference to the parent map
    // std::vector<NonObliviousOHashMap*> nonOMaps;  // non-oblivious maps
    std::vector<uint64_t> factors;  // number of ways in each level of butterfly
    Element* batch;                 // buffer for the current batches
    Element* tempElements;          // buffer for mergesplit
    uint8_t* tempMarks;             // buffer for mergesplit
    uint64_t shardCount;            // number of shards
    uint64_t bktSize;               // size of each bucket
    uint64_t bktPerBatch;           // number of buckets in each batch
    uint64_t bktRealSize;           // number of real elements in each bucket
    uint64_t parBatchCount;         // number of batches in parallel
    uint64_t batchSize;          // size of each batch (in number of elements)
    uint64_t perBatchShardSize;  // size of each shard in each batch
    uint64_t cacheBytes;         // cache size available
    uint64_t currBktIdx;         // current bucket index in the current batch
    uint64_t currBktOffset;      // current offset in the current bucket
    uint64_t load;       // number of elements inserted in the current batch
    int butterflyLevel;  // number of levels in the butterfly network

    /**
     * @brief Route elements in the current batch through the butterfly network,
     * and insert into the non oblivious map.
     *
     */
    void route() {
      // route the data through the butterfly network in parallel
      bool initFailed = false;
      for (int level = 0, stride = parBatchCount; level < butterflyLevel;
           ++level) {
        uint64_t way = factors[level];
        uint64_t parCount = bktPerBatch / way;
// #pragma omp parallel for schedule(static)
        for (uint64_t parIdx = 0; parIdx < parCount; ++parIdx) {
          uint64_t groupIdx = parIdx / stride;
          uint64_t groupOffset = parIdx % stride;
          Element* KWayIts[8];
          for (uint64_t j = 0; j < way; ++j) {
            KWayIts[j] =
                batch + ((j + groupIdx * way) * stride + groupOffset) * bktSize;
          }
          size_t tempBktsSize = way * bktSize * sizeof(Element);
          Element* localTempElements = tempElements + parIdx * way * bktSize;
          uint8_t* localTempMarks = tempMarks + parIdx * way * bktSize;
          try {
            MergeSplitKWay(KWayIts, way, bktSize, localTempElements,
                           localTempMarks);
          } catch (const std::runtime_error& e) {
            initFailed = true;
          }
        }
        stride *= way;
      }
      if (initFailed) {
        throw std::runtime_error(
            "Initialization failed due to bucket overflow. Please retry.");
      }
      // insert the data into the non-oblivious maps in parallel
// #pragma omp parallel for schedule(static)
    //   for (uint32_t i = 0; i < shardCount; ++i) {
    //     for (uint64_t j = 0; j < perBatchShardSize; ++j) {
    //       const Element& elem = batch[i * perBatchShardSize + j];
    //       const KVPair& kvPair = elem.v;
    //       // insert dummies like normal elements
    //       bool existFlag = nonOMaps[i]->template Insert<true>(
    //           kvPair.key, kvPair.value, elem.IsDummy());
    //       if (existFlag & !elem.IsDummy()) {
    //         initFailed = true;
    //         break;
    //       }
    //     }
    //   }
    //   if (initFailed) {
    //     throw std::runtime_error(
    //         "Initialization failed. Encountered duplicate keys.");
    //   }
    }

    /**
     * @brief Construct a new InitContext object
     *
     * @param omap The parent map
     * @param initSize The number of elements to be inserted
     * @param cacheBytes The cache size for all shards
     */
    InitContext(ParOMap& omap, PositionType initSize,
                uint64_t cacheBytes)
        : omap(omap),
          cacheBytes(cacheBytes),
          currBktIdx(0),
          currBktOffset(0),
          load(0) {
      if (omap.size() == 0) {
        throw std::runtime_error("ParOMap size not set. Call SetSize first.");
      }
      if (omap.inited) {
        throw std::runtime_error("ParOMap double initialization");
      }
      shardCount = omap.numShards;
      uint64_t maxInitSizePerShard =
          maxQueryPerShard(initSize, shardCount, -60);
    //   using NonObliviousOHashMap = OHashMap<K, V, false, PositionType>;
    //   nonOMaps.resize(shardCount);

      using Element = EM::Algorithm::TaggedT<KVPair>;
      bktSize = std::min(16384UL, GetNextPowerOfTwo(maxInitSizePerShard));
      bktRealSize = numRealPerBkt(bktSize, shardCount, -60);
      uint64_t minBatchSize = bktSize * shardCount;
      uint64_t maxBatchSize = cacheBytes / (sizeof(Element) + 8) / 2;
      factors = omap.factorizeShardCount(shardCount);
      if (maxBatchSize < minBatchSize) {
        throw std::runtime_error("InitFromReader cache size too small");
      }

      uint64_t totalBktNeeded = divRoundUp(initSize, bktRealSize);
      // make sure it's a multiple of shardCount
      uint64_t bktPerShard = divRoundUp(totalBktNeeded, shardCount);
      if (bktPerShard * bktSize > omap.shardSize) {
        omap.shardSize =
            bktPerShard *
            bktSize;  // we need large oram to hold these many elements
      }
    //   for (auto& nonOMapPtr : nonOMaps) {
    //     nonOMapPtr = new NonObliviousOHashMap();
    //     nonOMapPtr->SetSize(omap.shardSize, 0);
    //   }
      totalBktNeeded = bktPerShard * shardCount;
      bktRealSize = divRoundUp(
          initSize, totalBktNeeded);  // split initial elements evenly
      if (totalBktNeeded * bktSize < maxBatchSize) {
        maxBatchSize = totalBktNeeded * bktSize;  // don't waste space
      }
      parBatchCount = maxBatchSize / minBatchSize;
      bktPerBatch = parBatchCount * shardCount;
      batchSize = bktPerBatch * bktSize;
      perBatchShardSize = parBatchCount * bktSize;

      // buffers for mergesplit
      batch = new Element[batchSize];
      tempElements = new Element[batchSize];
      tempMarks = new uint8_t[batchSize];
      butterflyLevel = factors.size();
    }

    InitContext(const InitContext&) = delete;
    InitContext(InitContext&&) = default;

    /**
     * @brief Insert a new key-value pair into the batch. The key must be
     * unique. The running time of this function can be unstable.
     * The method may throw error with some negligible probability (when some
     * bucket overflows), in which case one may retry the initialization.
     *
     * @param key
     * @param value
     */
    void Insert(const K& key, const V& value) {
      if (load >= omap.size()) {
        throw std::runtime_error("Too many elements inserted during init");
      }
      ++load;
      Element& elem = batch[currBktIdx * bktSize + currBktOffset];
      elem.v.key = key;
      elem.v.value = value;
      uint64_t hash = secure_hash_with_salt((uint8_t*)&elem.v.key, sizeof(K),
                                           omap.randSalt);
      elem.setTag(omap.getShardByHash(hash));
      if (++currBktOffset == bktRealSize) {
        uint64_t baseOffset = currBktIdx * bktSize;
        for (; currBktOffset < bktSize; ++currBktOffset) {
          batch[baseOffset + currBktOffset].setDummy();
        }
        currBktOffset = 0;
        if (++currBktIdx == bktPerBatch) {
          route();
          currBktIdx = 0;
        }
      }
    }

    void Insert(const std::pair<K, V>& kvPair) { Insert(kvPair.first, kvPair.second); }

    template <class Iterator>
    void InsertBatch(Iterator begin, Iterator end) {
      for (auto it = begin; it != end; ++it) {
        Insert(it->first, it->second);
      }
    }

    /**
     * @brief Finished inserting elements. Close the context and initialize
     * the map. The method may throw error with some negligible probability,
     * in which case one may retry the initialization.
     *
     */
    void Finalize() {
      // set the rest of buckets to dummy
      uint64_t batchOffset = currBktIdx * bktSize + currBktOffset;
      if (batchOffset != 0) {
        for (uint64_t i = batchOffset; i < batchSize; ++i) {
          batch[i].setDummy();
        }
        route();
      }
      delete[] tempElements;
      delete[] tempMarks;
      delete[] batch;
      tempElements = NULL;
      tempMarks = NULL;
      batch = NULL;
//       for (auto& shard : omap.shards) {
//         shard.SetSize(omap.shardSize, cacheBytes / shardCount);
//       }
// // initialize the oblivious maps from the non-oblivious maps in parallel
// // #pragma omp parallel for schedule(static)
//       for (uint32_t i = 0; i < shardCount; ++i) {
//         omap.shards[i].InitFromNonOblivious(*nonOMaps[i]);
//         delete nonOMaps[i];
//         nonOMaps[i] = NULL;
//       }
    }

    ~InitContext() {
      if (tempElements) {
        delete[] tempElements;
      }
      if (tempMarks) {
        delete[] tempMarks;
      }
      if (batch) {
        delete[] batch;
      }
    //   for (auto& nonOMapPtr : nonOMaps) {
    //     if (nonOMapPtr) {
    //       delete nonOMapPtr;
    //     }
    //   }
    }
  };

  /**
   * @brief Obtain a new context to initialize this map. The initialization
   data
   * can be either private or public (the initialization and subsequent
   accesses
   * are oblivious). The data needs to be unique.
   * Example:
   *  auto* initContext = parOMap.NewInitContext(kvMap.size(), 1UL << 28);
      for (auto it = kvMap.begin(); it != kvMap.end(); ++it;) {
        initContext->Insert(it->first, it->second);
      }
      initContext->Finalize();
      delete initContext;
   *
   * @param initSize The number of elements to be inserted (could be an
   estimate, but should be no less than the actual number of elements)
   * @param cacheBytes The cache size for all shards.
   * @return InitContext The context object
   */
  InitContext* NewInitContext(PositionType initSize,
                              uint64_t cacheBytes) {
    return new InitContext(*this, initSize, cacheBytes);
  }

  /**
   * @brief Initialize the map from a reader. The data is fetched in batches
   * and for each batch, we route the data in parallel through a multi-way
   * butterfly network to load balance them to each shard. Might throw error with
   * some negligible probability (when some bucket overflows), in which case
   * one may retry.
   *
   * @tparam Reader The type of the reader
   * @param reader The reader object
   * @param cacheBytes The cache size for all shards
   */
  template <class Reader>
  void InitFromReader(Reader& reader, uint64_t cacheBytes) {
    InitContext context(*this, reader.size(), cacheBytes);
    while (!reader.eof()) {
      const std::pair<K, V>& kvPair = reader.read();
      context.Insert(kvPair.first, kvPair.second);
    }
    context.Finalize();
  }

  // As described in Snoopy paper.
  template <class Reader>
  void InitFromReaderBitonic(Reader& reader) {
    uint64_t inputSize = reader.size();
    const uint64_t bktSize = 16384; // corresponds to alpha in Snoopy paper
    const uint64_t bktRealSize = numRealPerBkt(bktSize, numShards);
    const uint64_t batchSize = bktRealSize * numShards;
    const uint64_t numBatch = divRoundUp(inputSize, batchSize);
    struct PartitionElement {
        K key;
        V value;
        uint32_t shardIdx;
        bool operator<(const PartitionElement& other) const {
            return shardIdx < other.shardIdx;
        }
    };
    
    for (uint64_t i = 0; i < numBatch; ++i) {
        uint64_t thisBatchSize = batchSize;
        if (i == numBatch - 1) {
            thisBatchSize = inputSize - i * batchSize;
        }
        StdVector<PartitionElement> vec(thisBatchSize + bktSize * numShards);
        for (uint64_t j = 0; j < thisBatchSize; ++j) {
            const std::pair<K, V>& kvPair = reader.read();
            uint64_t hash = secure_hash_with_salt((uint8_t*)&kvPair.first, sizeof(K), randSalt);
            vec[j] = PartitionElement{kvPair.first, kvPair.second, getShardByHash(hash)};
        }
        for (uint32_t j = 0; j < numShards; ++j) {
            for (uint64_t k = 0; k < bktSize; ++k) {
                vec[thisBatchSize + j * bktSize + k] = PartitionElement{K(), V(), j};
            }
        }
        EM::Algorithm::BitonicSort(vec.begin(), vec.end());
        std::vector<uint64_t> prefixSum(vec.size() + 1);
        uint32_t currShardIdx = 0;
        uint64_t currShardCount = 0;
        prefixSum[0] = 0;
        for (uint64_t j = 0; j < vec.size(); ++j) {
            bool isSameShard = vec[j].shardIdx == currShardIdx;
            obliMove(!isSameShard, currShardCount, 0UL);
            obliMove(!isSameShard, currShardIdx, vec[j].shardIdx);
            bool keepFlag = currShardCount < bktSize;
            ++currShardCount;
            prefixSum[j + 1] = prefixSum[j] + keepFlag;
        }
        EM::Algorithm::OrCompactSeparateMark(vec.begin(), vec.end(), prefixSum.begin());
    }
    
  }

  
};
}   // namespace Apps