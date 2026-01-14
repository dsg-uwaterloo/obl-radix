#pragma once

#include "apps_common.hpp"
namespace Apps {
using namespace EM::Algorithm;
struct ORAMEntry {
  uint64_t pos;
  uint64_t uid;
  // 32 key + 32 value stored in a cuckoo hash table with bucket size = 2
  Bytes<128> data;
};

/**
 * @brief Initialize the ORAM.
 * @tparam Z the bucket size of the ORAM
 * @tparam Reader the type of the reader
 * @tparam Writer the type of the writer
 * @param reader should read in ORAMEntries
 * @param writer should write out ORAMEntries in the bottomup order of the tree.
 * @param level the current level of the ORAM tree counting bottom up
 */
template <class IntermediateVec, const SortMethod method, uint64_t Z = 4,
          class Reader, class Writer>
void ORAMInitHelper(Reader& reader, Writer& writer, int level = 0) {
  static_assert(method == KWAYBUTTERFLYOSORT || method == BITONICSORT,
                "Invalid method for ORAMInit");
  uint64_t size = reader.size();
  if (size & (size - 1)) {
    throw std::runtime_error("size should be power of 2");
  }
  if (!size) {
    return;
  }
  struct LevelPosEntry {
    uint64_t levelPos;
    ORAMEntry entry;
    bool operator<(const LevelPosEntry& other) const {
      return (levelPos < other.levelPos) |
             (levelPos == other.levelPos && entry.uid < other.entry.uid);
    }
  };
  struct MarkEntry {
    bool isMarked;
    ORAMEntry entry;
    bool operator<(const MarkEntry& other) const {
      // put marked elements to the front and sort by pos
      // for unmarked, put dummy entries to the end
      return (isMarked > other.isMarked) |
             (other.isMarked & isMarked & (entry.pos < other.entry.pos)) |
             (!other.isMarked & !isMarked & (entry.uid < other.entry.uid));
    }
  };
  IntermediateVec upVec(size / 2);  // for the upper level
  typename IntermediateVec::Writer upWriter(upVec.begin(), upVec.end());
  {
    EM::VirtualVector::VirtualReader<LevelPosEntry> vReader(
        size * (Z + 1), [&](uint64_t idx) {
          if (idx >= size) {
            uint64_t levelPos = (idx - size) / Z;
            return LevelPosEntry{
                levelPos,
                ORAMEntry{levelPos << level, (uint64_t)-1, Bytes<128>{}}};
          }
          const ORAMEntry& entry = reader.read();
          return LevelPosEntry{entry.pos >> level, entry};
        });

    IntermediateVec intermediateVec(size * (Z + 1));
    typename IntermediateVec::Writer intermediateWriter(intermediateVec.begin(),
                                                        intermediateVec.end());
    EM::VirtualVector::VirtualWriter<LevelPosEntry> intermediateVWriter(
        size * (Z + 1), [&](uint64_t idx, const LevelPosEntry& entry) {
          intermediateWriter.write(entry.entry);
        });
    uint64_t heapSize = DEFAULT_HEAP_SIZE;
    if constexpr (std::is_same<IntermediateVec, StdVector<ORAMEntry>>::value) {
      heapSize -= sizeof(ORAMEntry) * size * (Z + 2);
    }
    if constexpr (method == KWAYBUTTERFLYOSORT) {
      KWayButterflySort(vReader, intermediateVWriter, heapSize);
    } else if constexpr (method == BITONICSORT) {
      BitonicSortRW(vReader, intermediateVWriter);
    }
    intermediateWriter.flush();
    typename IntermediateVec::Reader intermediateReader(intermediateVec.begin(),
                                                        intermediateVec.end());
    uint64_t prevPos = (uint64_t)-1;
    uint64_t dupCounter = 0;
    EM::VirtualVector::VirtualReader<MarkEntry> vReader2(
        size * (Z + 1), [&](uint64_t idx) {
          const ORAMEntry& entry = intermediateReader.read();
          bool isDup = (entry.pos >> level) == (prevPos >> level);
          obliMove(!isDup, dupCounter, 0UL);

          bool isMarked = dupCounter < Z;
          ++dupCounter;
          prevPos = entry.pos;
          return MarkEntry{isMarked, entry};
        });

    EM::VirtualVector::VirtualWriter<MarkEntry> vWriter(
        size * (Z + 1), [&](uint64_t idx, const MarkEntry& entry) {
          if (idx < size * Z) {
            writer.write(entry.entry);
          } else if (idx < size * Z + size / 2) {
            upWriter.write(entry.entry);
          }
        });
    if constexpr (method == KWAYBUTTERFLYOSORT) {
      KWayButterflySort(vReader2, vWriter, heapSize);
    } else if constexpr (method == BITONICSORT) {
      BitonicSortRW(vReader2, vWriter);
    }
    upWriter.flush();
  }
  typename IntermediateVec::Reader upReader(upVec.begin(), upVec.end());
  ORAMInitHelper<IntermediateVec, method, Z>(upReader, writer, level + 1);
}

/**
 * @brief Initialize the ORAM.
 * @tparam Z the bucket size of the ORAM
 * @tparam Reader the type of the reader
 * @tparam Writer the type of the writer
 * @param reader should read in ORAMEntries
 * @param writer should write out ORAMEntries in the bottomup order of the tree.
 */
template <const SortMethod method, uint64_t Z = 4, class Reader, class Writer>
void ORAMInit(Reader& reader, Writer& writer) {
  static_assert(method == KWAYBUTTERFLYOSORT || method == BITONICSORT,
                "Invalid method for ORAMInit");
  if (DEFAULT_HEAP_SIZE < sizeof(ORAMEntry) * reader.size() * (Z * 5 + 2)) {
    printf("Uses external memory\n");
    using IntermediateVec = EM::NonCachedVector::Vector<ORAMEntry>;
    ORAMInitHelper<IntermediateVec, method, Z>(reader, writer);
  } else {
    using IntermediateVec = StdVector<ORAMEntry>;
    ORAMInitHelper<IntermediateVec, method, Z>(reader, writer);
  }
}
}  // namespace Apps