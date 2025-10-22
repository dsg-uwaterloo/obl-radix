#pragma once
#include <cstdint>

// triple32 hash function: https://github.com/skeeto/hash-prospector
// exact bias: 0.020888578919738908
inline uint32_t triple32(uint32_t x) {
  x ^= x >> 17;
  x *= 0xed5ad4bb;
  x ^= x >> 11;
  x *= 0xac4c1b51;
  x ^= x >> 15;
  x *= 0x31848bab;
  x ^= x >> 14;
  return x;
}
