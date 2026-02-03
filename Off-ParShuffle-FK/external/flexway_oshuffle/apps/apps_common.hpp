#pragma once
#include "external_memory/algorithm/bitonic.hpp"
#include "external_memory/algorithm/kway_butterfly_sort.hpp"
#include "external_memory/stdvector.hpp"
#include "external_memory/virtualvector.hpp"
using namespace EM::Algorithm;
template <class SortVec, class Reader, class Writer>
void BitonicSortRWHelper(Reader& reader, Writer& writer) {
  uint64_t size = reader.size();

  SortVec vec(size);
  typename SortVec::Writer vecWriter(vec.begin(), vec.end());
  while (!reader.eof()) {
    vecWriter.write(reader.read());
  }
  BitonicSort(vec.begin(), vec.end());
  typename SortVec::Reader vecReader(vec.begin(), vec.end());
  while (!vecReader.eof()) {
    writer.write(vecReader.read());
  }
}

template <class Reader, class Writer>
void BitonicSortRW(Reader& reader, Writer& writer) {
  uint64_t size = reader.size();
  using T = typename Reader::value_type;
  if (DEFAULT_HEAP_SIZE < sizeof(T) * size * 2) {
    static constexpr uint64_t page_size = 1UL << 16;
    static constexpr uint64_t cache_size = DEFAULT_HEAP_SIZE / page_size;
    using SortVec = EM::ExtVector::Vector<T, page_size, true, true, cache_size>;
    BitonicSortRWHelper<SortVec>(reader, writer);
  } else {
    using SortVec = StdVector<T>;
    BitonicSortRWHelper<SortVec>(reader, writer);
  }
}