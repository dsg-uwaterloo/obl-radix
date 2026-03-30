#pragma once
// This header is intentionally lightweight; it relies on INLINE/Assert macros
// provided by the enclosing oshuffle codebase.
#include <algorithm>
#include <bit>
#include <cstdint>
#include <type_traits>
/// @brief EdgeRec is a bitset that records the edges in a graph
/// @tparam Bits the type of the bitset, default is uint64_t
template <typename Bits = uint64_t>
struct EdgeRec {
  Bits edges;
  uint16_t k;

 private:
  // Constant-time helpers: avoid data-dependent control flow while selecting
  // values derived from secret randomness (shuffle tags).
  INLINE static constexpr uint64_t ctMaskNonZero(uint64_t x) {
    return 0ULL - (uint64_t)(x != 0);
  }
  INLINE static constexpr uint64_t ctSelect(uint64_t mask, uint64_t a,
                                           uint64_t b) {
    return (a & mask) | (b & ~mask);
  }
  INLINE static constexpr uint16_t ctSelect16(uint64_t mask, uint16_t a,
                                             uint16_t b) {
    const uint16_t m16 = (uint16_t)mask;
    return (uint16_t)((a & m16) | (b & (uint16_t)~m16));
  }

 public:
  EdgeRec(uint16_t k) : k(k) {  // k is number of vertices
    static_assert(std::is_same<Bits, uint64_t>::value,
                  "unrecognized type for recording edge");
    edges = 0;
  }
  // map vertex to edge offset in Edge Rec
  // it's ordered by 0-0, 1-0, 1-1, 2-0, 2-1, ...
  INLINE static constexpr uint16_t getEdgeOffset(uint16_t v0, uint16_t v1) {
    return (v0 << 3) | v1;
  }

  INLINE static constexpr Bits getEdgesMask(uint16_t v0, uint16_t v1) {
    if constexpr (std::is_same<Bits, uint64_t>::value) {
      return (1UL << getEdgeOffset(v0, v1)) | (1UL << getEdgeOffset(v1, v0));
    }
    return 0;
  }

  INLINE void flipEdge(Bits mask) {
    if constexpr (std::is_same<Bits, uint64_t>::value) {
      edges ^= mask;
    }
  }

  INLINE void flipEdge(uint16_t v0, uint16_t v1) {
    flipEdge(getEdgesMask(v0, v1));
  }

  INLINE bool retrieveEdge(uint16_t edgeOffset) const {
    if constexpr (std::is_same<Bits, uint64_t>::value) {
      return (edges >> edgeOffset) & 1UL;
    }
  }

  INLINE bool retrieveEdge(uint16_t v0, uint16_t v1) const {
    uint16_t offset = getEdgeOffset(v0, v1);
    return retrieveEdge(offset);
  }

  // retrieve the direction of v0->v1 and flip the direction
  INLINE bool retrieveAndFlipEdge(uint16_t v0, uint16_t v1) {
    Bits mask = getEdgesMask(v0, v1);
    Bits v0_to_v1_mask = 1UL << getEdgeOffset(v0, v1);
    Bits v1_to_v0_mask = 1UL << getEdgeOffset(v1, v0);
    edges ^= v0_to_v1_mask | v1_to_v0_mask;
    return !(edges & v0_to_v1_mask);
  }

  void print() {
    for (uint16_t i = 0; i < k; ++i) {
      for (uint16_t j = i + 1; j < k; ++j) {
        if (retrieveEdge(i, j)) {
          printf("%d - %d\n", i, j);
        }
      }
    }
  }

  void printPath(const EdgeRec& path) {
    Assert(k == path.k);
    for (uint16_t i = 0; i < k; ++i) {
      for (uint16_t j = i + 1; j < k; ++j) {
        if (retrieveEdge(i, j)) {
          if (path.retrieveEdge(i, j)) {
            printf("%d -> %d\n", i, j);
          } else {
            printf("%d -> %d\n", j, i);
          }
        }
      }
    }
  }

  INLINE EdgeRec EulerPath(uint64_t numEdge) {
    numEdge = std::min(numEdge, (uint64_t)(k * (k - 1) / 2));
    // an edge is set to 1 if it's direction is from small vertex to larger
    EdgeRec path(k);
    // for edges appearing even number of times, set direction according to
    // their order
    path.edges = (~edges) & 0xFF7F3F1F0F070301UL;
    static constexpr Bits simpleMask = ~0x8040201008040201UL;
    // filtering out edges pointing to itself
    edges &= simpleMask;
    uint16_t v8 = 0;
    for (uint16_t r = 0; r != numEdge; ++r) {
      // Maintain data-independent control flow: do not branch on `edges` (it is
      // derived from secret shuffle tags). Instead, compute masks and apply
      // updates conditionally.
      const uint64_t edgesMask = ctMaskNonZero((uint64_t)edges);

      // Clear bits for less than v0; if none remain, wrap around to the global
      // smallest uncovered vertex (case 2 in comments below).
      const uint64_t choices0 = (uint64_t)edges & ((-1ULL) << v8);
      const uint64_t choicesMask = ctMaskNonZero(choices0);
      const uint64_t choicesWrapped = ctSelect(choicesMask, choices0, (uint64_t)edges);

      // Avoid shift-by-64 UB: ensure the value passed into countr_zero is never
      // zero when `edges` is empty. Subsequent updates are masked off anyway.
      const uint64_t safeChoices =
          choicesWrapped | ((~edgesMask) & 1ULL);  // if edges==0 -> 1
      const uint16_t trailingZeros = (uint16_t)std::countr_zero(safeChoices);
      // case 1: there's an edge v0->next
      // case 2: next is a vertex connected to the smallest uncovered vertex
      // case 3: all is done and next = 0
      uint16_t next = trailingZeros & 7;
      uint16_t v0 = trailingZeros >> 3;
      const Bits v0_to_next_mask = (Bits)(1ULL << trailingZeros);
      const uint16_t nextV8 = (uint16_t)(next << 3);
      const Bits next_to_v0_mask = (Bits)(1ULL << (uint16_t)(nextV8 | v0));
      // set this edge as done (could be that it's already done in case 2)
      edges &= ~(Bits)((v0_to_next_mask | next_to_v0_mask) & (Bits)edgesMask);
      // set direction v0->next if it's in case 1
      path.edges |= (Bits)(v0_to_next_mask & (Bits)edgesMask);
      v8 = ctSelect16(edgesMask, nextV8, v8);
    }
    return path;
  }
};
