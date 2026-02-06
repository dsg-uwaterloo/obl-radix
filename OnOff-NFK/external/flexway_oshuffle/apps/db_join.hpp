#pragma once
#include "apps_common.hpp"

namespace Apps {
using namespace EM::Algorithm;
template <typename T>
struct DBEntry {
  T payload;
  uint64_t id;
  bool operator<(const DBEntry& other) const {
    return id < other.id;
  }
  bool operator==(const DBEntry& other) const {
    return id == other.id;
  }
};

template <typename DBEntry_>
struct SortEntry {
    DBEntry_ entry;
    uint8_t tableId;
    bool operator<(const SortEntry& other) const {
        return (entry < other.entry) | ((entry == other.entry) & (tableId > other.tableId));
    }
};

template <const SortMethod method, class IntermediateVec, class Reader1, class Reader2,
          class Writer>
void dbJoinHelper(Reader1& reader1, Reader2& reader2, Writer& writer) {
  using DBEntry1 = typename Reader1::value_type;
  using DBEntry2 = typename Reader2::value_type;
  using DBEntry_ = typename Writer::value_type;
  using SortEntry_ = SortEntry<DBEntry_>;
  uint64_t size1 = reader1.size();
  uint64_t size2 = reader2.size();
  uint64_t totalSize = size1 + size2;
  
  IntermediateVec db(totalSize);
  typename EM::VirtualVector::VirtualReader<SortEntry_> vReader(totalSize, [&](uint64_t idx) {
    SortEntry_ ele;
    if (idx < size1) {
      ele.tableId = 1;
      const auto& readEntry = reader1.read();
      ele.entry.payload.first = readEntry.payload;
      ele.entry.id = readEntry.id;
    } else {
      ele.tableId = 2;
      const auto& readEntry = reader2.read();
      ele.entry.payload.second = readEntry.payload;
      ele.entry.id = readEntry.id;
    }
    return ele;
  });
  typename IntermediateVec::Writer dbWriter(db.begin(), db.end());
  SortEntry_ prev;
  EM::VirtualVector::VirtualWriter<SortEntry_> vWriter(
      totalSize, [&](uint64_t idx, const SortEntry_& ele) {
        SortEntry_ out = ele;
        if (idx == 0) {
          dbWriter.write(out);
          prev = ele;
          return;
        }
        bool isDup = prev.entry.id == ele.entry.id;
        obliMove(isDup, out.entry.payload.second, prev.entry.payload.second);
        dbWriter.write(out);
        prev = ele;
      });
    EM::VirtualVector::VirtualWriter<SortEntry_> vOutWriter(
        totalSize, [&](uint64_t idx, const SortEntry_& ele) {
        writer.write(ele.entry);
    });
  uint64_t heapSize = DEFAULT_HEAP_SIZE;
  if constexpr (std::is_same<IntermediateVec, StdVector<SortEntry_>>::value) {
    heapSize -= sizeof(SortEntry_) * totalSize;
  }
  if constexpr (method == KWAYBUTTERFLYOSORT) {
    KWayButterflySort(vReader, vWriter, heapSize);
    dbWriter.flush();
    // std::cout << "write " << prev << " " << dupCounter << std::endl;
    typename IntermediateVec::Reader dbReader(db.begin(), db.end());
    KWayButterflySort(dbReader, vOutWriter, heapSize);
  } else if constexpr (method == BITONICSORT) {
    BitonicSortRW(vReader, vWriter);
    dbWriter.flush();
    // std::cout << "write " << prev << " " << dupCounter << std::endl;
    typename IntermediateVec::Reader dbReader(db.begin(), db.end());
    BitonicSortRW(dbReader, vOutWriter);
  }
}
template <const SortMethod method, class Reader1, class Reader2, class Writer>
void dbJoin(Reader1& reader1, Reader2& reader2, Writer& writer) {
  uint64_t size = reader1.size() + reader2.size();
  using DBEntry_ = typename Writer::value_type;
  using SortEntry_ = SortEntry<DBEntry_>;
  static_assert(method == KWAYBUTTERFLYOSORT || method == BITONICSORT,
                "Invalid method for dbJoin");

  if (DEFAULT_HEAP_SIZE < sizeof(SortEntry_) * size * 5) {  // external memory
    using IntermediateVec =
        EM::NonCachedVector::Vector<SortEntry_, (1UL << 16)>;
    printf("Uses external memory\n");
    dbJoinHelper<method, IntermediateVec>(reader1, reader2, writer);
  } else {
    using IntermediateVec = StdVector<SortEntry_>;
    dbJoinHelper<method, IntermediateVec>(reader1, reader2, writer);
  }
}
}  // namespace Apps