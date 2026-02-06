#pragma once
#include "common/defs.hpp"
#include "common/dummy.hpp"
#include "common/encrypted.hpp"
#include "common/mov_intrinsics.hpp"
#ifdef ENCLAVE_MODE
#include "sgx_eid.h"
#include "sgx_tcrypto.h"
#include "sgx_trts.h"
#include "sgx_thread.h"
#include "sgx_tseal.h"
#endif
#include <omp.h>

// size of each element in bytes
#ifndef ELEMENT_SIZE
#define ELEMENT_SIZE 128
#endif

#define MAX_THREAD_COUNT omp_get_max_threads()
int thread_count = MAX_THREAD_COUNT;

namespace EM::Algorithm {
enum SortMethod {
  CABUCKETSORT,
  BITONICSORT,
  ORSHUFFLE,
  CABUCKETSHUFFLE,
  BITONICSHUFFLE,
  KWAYDISTRIBUTIONOSORT,
  KWAYDISTRIBUTIONOSORTSHUFFLED,
  KWAYBUTTERFLYOSORT,
  KWAYBUTTERFLYOSHUFFLE,
  UNOPTBITONICSORT,
  EXTMERGESORT,
  OTHER
};

enum PartitionMethod {
  INTERLEAVE_PARTITION,
  OR_COMPACT,
  GOODRICH_COMPACT,
  BITONIC
};

template <bool perf = true>
INLINE void condSwap(const auto &cond, auto &v1, auto &v2) {
  if constexpr (perf) {
    PERFCTR_INCREMENT(swapCount);
  }
  obliSwap(cond, v1, v2);
}

template <bool perf = true>
INLINE void swap(auto &v1, auto &v2) {
  if constexpr (perf) {
    PERFCTR_INCREMENT(swapCount);
  }
  std::swap(v1, v2);
}

template <template <typename> class Vector2, typename T, typename Compare>
bool IsSorted(Vector2<T> &v, Compare cmp) {
  bool ret = true;
  for (uint64_t i = 1; i < v.size(); i++) {
    ret = ret * (cmp(v[i - 1], v[i]) + (!cmp(v[i], v[i - 1])));
  }
  return ret;
}

template <typename T>
  requires(IS_POD<T>())
struct Block {
#if defined(__AVX512VL__) || defined(__AVX2__)
  static constexpr size_t paddingSize = sizeof(T) % 32 == 16 ? 8 : 0;
#else
  static constexpr size_t paddingSize = 0;
#endif
  T data;
  uint32_t tag;
  bool dummyFlag;
  bool lessFlag;
  char padding[paddingSize];
  auto operator==(const Block &other) const { return data == other.data; }
  bool operator<(const Block &other) const {
    return (data < other.data) | ((data == other.data) & (tag < other.tag));
  }
  inline void setData(const T &_data) {
    data = _data;
    tag = UniformRandom32();
    dummyFlag = false;
  }

  inline void setData(const T &_data, uint32_t i) {
    data = _data;
    tag = i;
    dummyFlag = false;
  }

  inline const T &getData() const { return data; }

  static consteval inline Block DUMMY() { return Block{T::DUMMY(), 0, true, false}; }
  inline bool isDummy() const { return dummyFlag; }
  inline bool setAndGetMarked(const Block &pivot) { return lessFlag = !(pivot < *this); }
  inline bool isMarked(const Block &unused) const { return lessFlag; }

  inline bool isLess() const { return lessFlag; }
  inline void setLessFlag(bool flag) { this->lessFlag = flag; }
  inline void condChangeMark(bool cond, const Block &unused) { this->lessFlag ^= cond; }
  inline void setDummy() { setDummyFlag(true); }
  inline void setDummyFlag(bool flag) { this->dummyFlag = flag; }
  inline void setDummyFlagCond(bool cond, bool flag) { obliMove(cond, this->dummyFlag, flag); }
};

template <typename T>
  requires(IS_POD<T>())
struct TaggedT {
#if defined(__AVX512VL__) || defined(__AVX2__)
  static constexpr size_t paddingSize = sizeof(T) % 32 == 16 ? 8 : 0;
#else
  static constexpr size_t paddingSize = 0;
#endif
  uint64_t tag; // high bit: dummy flag
  T v;
  char padding[paddingSize];

  inline void setData(const T &_data) {
    v = _data;
    tag = UniformRandom() & 0x7fff'ffff'ffff'ffffUL;
    if (tag == 0) {
      printf("UniformRandom() returns 0\n");
    }
  }

  inline void setData(const T &_data, RandGen &custom_rand) {
    v = _data;
    tag = custom_rand.rand64() & 0x7fff'ffff'ffff'ffffUL;
  }

  inline const T &getData() const { return v; }

  inline bool isDummy() const { return tag >> 63; }

  inline void setDummy() { tag |= 0x8000'0000'0000'0000UL; }

  inline void setTag(uint64_t _tag) { tag = _tag & 0x7fff'ffff'ffff'ffffUL; }

  inline bool setAndGetMarked(uint64_t bitMask) const { return isMarked(bitMask); }

  inline bool isMarked(uint64_t bitMask) const { return !(tag & bitMask); }

  inline void condChangeMark(bool cond, uint64_t bitMask) {
    uint64_t newTag = tag ^ bitMask;
    obliMove(cond, tag, newTag);
  }

  inline uint8_t getMarkAndUpdate(uint64_t k) {
    uint64_t realTag = tag & 0x7fff'ffff'ffff'ffffUL;
    tag &= 0x8000'0000'0000'0000UL;
    tag |= realTag / k;
    uint8_t mark = realTag % k;
    return mark;
  }
};
} // namespace EM::Algorithm

template <typename T>
INLINE void CMOV(const uint64_t &condition, T &A, const T &B) {
  obliMove(condition != 0, A, B);
}

template <typename T>
INLINE void CMOV(const uint64_t &condition, EM::Algorithm::Block<T> &A,
                 const EM::Algorithm::Block<T> &B) {
  CMOV(condition, A.data, B.data);
  CMOV(condition, A.tag, B.tag);
  CMOV(condition, A.dummyFlag, B.dummyFlag);
  CMOV(condition, A.lessFlag, B.lessFlag);
}

template <typename T>
INLINE void CMOV(const uint64_t &condition, EM::Algorithm::TaggedT<T> &A,
                 const EM::Algorithm::TaggedT<T> &B) {
  CMOV(condition, A.tag, B.tag);
  CMOV(condition, A.v, B.v);
}
