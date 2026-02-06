// triple32:
// // #pragma once
// #include "data-types.h"
// #include "inputs.h"
// #include "slice_utils.h"
// #include "triple32.h"
// #include <cstdint>
// #include <cstring>
// #include <immintrin.h>
// #include <thread>
// #include <vector>

// using std::vector;

// inline uint32_t generateDummy(uint32_t key, uint32_t index) {
//   return triple32(key ^ index);
// }

// inline void replaceWithDummiesParallel(table_t &table,
//                                        const vector<Slice> &slices)

// {
//   const uint32_t P = slices.size();
//   vector<std::thread> pool;
//   pool.reserve(P);

//   for (uint32_t t = 0; t < P; ++t) {
//     pool.emplace_back([&, t] {
//       const Slice sl = slices[t];
//       uint32_t i = sl.begin;
//       const uint32_t end = sl.end;

//       for (; i + 4 <= end; i += 4) {
//         if (i + 32 < end) {
//           _mm_prefetch(reinterpret_cast<const char *>(&table.tuples[i + 32]),
//                        _MM_HINT_T0);
//         }

//         uint32_t d0 = generateDummy(table.tuples[i].key, i);
//         uint32_t d1 = generateDummy(table.tuples[i + 1].key, i + 1);
//         uint32_t d2 = generateDummy(table.tuples[i + 2].key, i + 2);
//         uint32_t d3 = generateDummy(table.tuples[i + 3].key, i + 3);

//         uint32_t m0 = -(table.tuples[i].cntSelf == 0);
//         uint32_t m1 = -(table.tuples[i + 1].cntSelf == 0);
//         uint32_t m2 = -(table.tuples[i + 2].cntSelf == 0);
//         uint32_t m3 = -(table.tuples[i + 3].cntSelf == 0);

//         table.tuples[i].idx = i;
//         table.tuples[i + 1].idx = i + 1;
//         table.tuples[i + 2].idx = i + 2;
//         table.tuples[i + 3].idx = i + 3;

//         table.tuples[i].hashKey = triple32((table.tuples[i].key & ~m0) | (d0
//         & m0)); table.tuples[i + 1].hashKey = triple32((table.tuples[i +
//         1].key & ~m1) | (d1 & m1)); table.tuples[i + 2].hashKey =
//         triple32((table.tuples[i + 2].key & ~m2) | (d2 & m2)); table.tuples[i
//         + 3].hashKey = triple32((table.tuples[i + 3].key & ~m3) | (d3 & m3));
//       }

//       for (; i < end; ++i) {
//         table.tuples[i].idx = i;
//         uint32_t dummy = generateDummy(table.tuples[i].key, i);
//         uint32_t mask = -(table.tuples[i].cntSelf == 0);
//         table.tuples[i].hashKey = triple32((table.tuples[i].key & ~mask) |
//         (dummy & mask));
//       }
//     });
//   }
//   for (auto &th : pool)
//     th.join();
// }

// intrinsics-based AES
// #pragma once

// #include <cstdint>
// #include <immintrin.h> // AES-NI + prefetch
// #include <stdexcept>
// #include <sys/random.h>
// #include <thread>
// #include <vector>

// #include "inputs.h" // Record / Slice definitions

// using std::vector;

// /* ───────────────────── Generate 128-bit random master key ───────────── */
// inline __m128i generateRandomAESKey() {
//   alignas(16) __m128i key; // 128-bit aligned buffer
//   ssize_t ret = getrandom(&key, sizeof(key), 0);
//   if (ret != sizeof(key))
//     throw std::runtime_error("getrandom failed to generate AES master key");
//   return key;
// }

// /* ───────────────────── AES-NI PRF returning 32-bit ───────────────────── */
// class AESPRF32 {
// private:
//   static inline __m128i expand(__m128i keygen, __m128i prev) {
//     keygen = _mm_shuffle_epi32(keygen, 0xff);
//     prev = _mm_xor_si128(prev, _mm_slli_si128(prev, 4));
//     prev = _mm_xor_si128(prev, _mm_slli_si128(prev, 4));
//     prev = _mm_xor_si128(prev, _mm_slli_si128(prev, 4));
//     return _mm_xor_si128(prev, keygen);
//   }

// public:
//   __m128i round_keys[11]; // AES-128: 10 rounds + initial key
//   explicit AESPRF32(__m128i master_key) {
//     round_keys[0] = master_key;
//     __m128i tmp = master_key;

//     tmp = _mm_aeskeygenassist_si128(tmp, 0x01);
//     round_keys[1] = expand(tmp, round_keys[0]);
//     tmp = _mm_aeskeygenassist_si128(tmp, 0x02);
//     round_keys[2] = expand(tmp, round_keys[1]);
//     tmp = _mm_aeskeygenassist_si128(tmp, 0x04);
//     round_keys[3] = expand(tmp, round_keys[2]);
//     tmp = _mm_aeskeygenassist_si128(tmp, 0x08);
//     round_keys[4] = expand(tmp, round_keys[3]);
//     tmp = _mm_aeskeygenassist_si128(tmp, 0x10);
//     round_keys[5] = expand(tmp, round_keys[4]);
//     tmp = _mm_aeskeygenassist_si128(tmp, 0x20);
//     round_keys[6] = expand(tmp, round_keys[5]);
//     tmp = _mm_aeskeygenassist_si128(tmp, 0x40);
//     round_keys[7] = expand(tmp, round_keys[6]);
//     tmp = _mm_aeskeygenassist_si128(tmp, 0x80);
//     round_keys[8] = expand(tmp, round_keys[7]);
//     tmp = _mm_aeskeygenassist_si128(tmp, 0x1B);
//     round_keys[9] = expand(tmp, round_keys[8]);
//     tmp = _mm_aeskeygenassist_si128(tmp, 0x36);
//     round_keys[10] = expand(tmp, round_keys[9]);
//   }

//   inline uint32_t operator()(uint64_t lo, uint64_t hi) const {
//     __m128i state = _mm_set_epi64x(hi, lo);
//     state = _mm_xor_si128(state, round_keys[0]);
//     for (int i = 1; i < 10; ++i)
//       state = _mm_aesenc_si128(state, round_keys[i]);
//     state = _mm_aesenclast_si128(state, round_keys[10]);
//     return _mm_cvtsi128_si32(state);
//   }
// };

// /* ───────────────────── 4-row vectorized PRF ─────────────────────────── */
// inline void prf4x32(const AESPRF32 &prf, uint64_t k0, uint64_t k1, uint64_t
// k2,
//                     uint64_t k3, uint32_t &out0, uint32_t &out1, uint32_t
//                     &out2, uint32_t &out3) {
//   __m128i block01 = _mm_set_epi64x(k1, k0);
//   __m128i block23 = _mm_set_epi64x(k3, k2);

//   block01 = _mm_xor_si128(block01, prf.round_keys[0]);
//   block23 = _mm_xor_si128(block23, prf.round_keys[0]);

//   for (int r = 1; r < 10; ++r) {
//     block01 = _mm_aesenc_si128(block01, prf.round_keys[r]);
//     block23 = _mm_aesenc_si128(block23, prf.round_keys[r]);
//   }

//   block01 = _mm_aesenclast_si128(block01, prf.round_keys[10]);
//   block23 = _mm_aesenclast_si128(block23, prf.round_keys[10]);

//   out0 = _mm_cvtsi128_si32(block01);
//   out1 = _mm_cvtsi128_si32(_mm_unpackhi_epi64(block01, block01));
//   out2 = _mm_cvtsi128_si32(block23);
//   out3 = _mm_cvtsi128_si32(_mm_unpackhi_epi64(block23, block23));
// }

// inline void replaceWithDummiesParallel(table_t &table,
//                                        const vector<Slice> &slices,
//                                        const __m128i &secret) {
//   constexpr uint64_t kHashSalt = 0xFFFFFFFFFFFFFFFFULL;
//   const uint32_t P = slices.size();
//   vector<std::thread> pool;
//   pool.reserve(P);

//   // // Generate a secure random master key once
//   // static const __m128i MASTER_KEY = generateRandomAESKey();

//   for (uint32_t t = 0; t < P; ++t) {
//     pool.emplace_back([&, t] {
//       // Thread-local AESPRF32 instance
//       thread_local AESPRF32 prf(secret);

//       const Slice sl = slices[t];
//       uint32_t i = sl.begin;
//       const uint32_t end = sl.end;

//       // Vectorized loop: 4 rows at a time
//       for (; i + 4 <= end; i += 4) {
//         if (i + 32 < end)
//           _mm_prefetch(reinterpret_cast<const char *>(&table.tuples[i + 32]),
//                        _MM_HINT_T0);

//         uint32_t d0, d1, d2, d3;
//         prf4x32(prf, table.tuples[i].key, table.tuples[i + 1].key,
//                 table.tuples[i + 2].key, table.tuples[i + 3].key, d0, d1, d2,
//                 d3);

//         uint32_t m0 = -(table.tuples[i].cntSelf == 0);
//         uint32_t m1 = -(table.tuples[i + 1].cntSelf == 0);
//         uint32_t m2 = -(table.tuples[i + 2].cntSelf == 0);
//         uint32_t m3 = -(table.tuples[i + 3].cntSelf == 0);

//         table.tuples[i].idx = i;
//         table.tuples[i + 1].idx = i + 1;
//         table.tuples[i + 2].idx = i + 2;
//         table.tuples[i + 3].idx = i + 3;

//         uint32_t k0 = (table.tuples[i].key & ~m0) | (d0 & m0);
//         uint32_t k1 = (table.tuples[i + 1].key & ~m1) | (d1 & m1);
//         uint32_t k2 = (table.tuples[i + 2].key & ~m2) | (d2 & m2);
//         uint32_t k3 = (table.tuples[i + 3].key & ~m3) | (d3 & m3);

//         table.tuples[i].hashKey = prf(k0, kHashSalt);
//         table.tuples[i + 1].hashKey = prf(k1, kHashSalt);
//         table.tuples[i + 2].hashKey = prf(k2, kHashSalt);
//         table.tuples[i + 3].hashKey = prf(k3, kHashSalt);
//       }

//       // Tail loop
//       for (; i < end; ++i) {
//         table.tuples[i].idx = i;
//         uint32_t dummy = prf(table.tuples[i].key, i);
//         uint32_t mask = -(table.tuples[i].cntSelf == 0);
//         uint32_t finalKey = (table.tuples[i].key & ~mask) | (dummy & mask);
//         table.tuples[i].hashKey = prf(finalKey, kHashSalt);
//       }
//     });
//   }

//   for (auto &th : pool)
//     th.join();
// }

// -----------------------------------------------------------------------------
//  fast_prf_replace_schemeA.hpp – AVX2, two-stage PRF (dummy + bucket hash)
// -----------------------------------------------------------------------------
#pragma once
#include <cstdint>
#include <cstring>
#include <immintrin.h> // AVX2 intrinsics + _mm_prefetch
#include <openssl/aes.h>
#include <thread>
#include <vector>

#include "data-types.h"
#include "inputs.h" // Record / Slice definitions
#include "slice_utils.h"

using std::vector;

/* ─────────────────────────── AES-based PRF helper ───────────────────────── */
class PRF_AES128_fast {
public:
  explicit PRF_AES128_fast(const __m128i &secret) {
    AES_set_encrypt_key(reinterpret_cast<const uint8_t *>(&secret), 128,
                        &sched_);
  }

  inline __m128i encrypt_block(__m128i block) const {
    const __m128i *rk =
        reinterpret_cast<const __m128i *>(static_cast<const void *>(sched_.rd_key));
    block = _mm_xor_si128(block, rk[0]);
    for (int round = 1; round < sched_.rounds; ++round)
      block = _mm_aesenc_si128(block, rk[round]);
    block = _mm_aesenclast_si128(block, rk[sched_.rounds]);
    return block;
  }

  /* one-block helper */
  inline uint64_t operator()(uint32_t key32, uint64_t salt64) const {
    __m128i block =
        _mm_set_epi64x(static_cast<long long>(salt64),
                       static_cast<long long>(static_cast<uint64_t>(key32)));
    __m128i enc = encrypt_block(block);
    return static_cast<uint64_t>(_mm_cvtsi128_si64(enc));
  }

  inline void operator()(const uint32_t key32[4], const uint64_t salt64[4],
                         uint64_t out[4]) const {
    __m128i state[4];
    for (int j = 0; j < 4; ++j) {
      state[j] = _mm_set_epi64x(static_cast<long long>(salt64[j]),
                                static_cast<long long>(
                                    static_cast<uint64_t>(key32[j])));
    }
    const __m128i *rk =
        reinterpret_cast<const __m128i *>(static_cast<const void *>(sched_.rd_key));
    for (int j = 0; j < 4; ++j)
      state[j] = _mm_xor_si128(state[j], rk[0]);
    for (int round = 1; round < sched_.rounds; ++round)
      for (int j = 0; j < 4; ++j)
        state[j] = _mm_aesenc_si128(state[j], rk[round]);
    for (int j = 0; j < 4; ++j)
      state[j] = _mm_aesenclast_si128(state[j], rk[sched_.rounds]);
    for (int j = 0; j < 4; ++j)
      out[j] = static_cast<uint64_t>(_mm_cvtsi128_si64(state[j]));
  }

  const AES_KEY &sched() const { return sched_; }

private:
  AES_KEY sched_;
};

/* ─────────────────────────── main routine (scheme A) ───────────────────────
 */
inline void replaceWithDummiesParallel(table_t &table,
                                       const vector<Slice> &slices,
                                       const __m128i &secret)

{
  constexpr uint64_t kHashSalt = 0xFFFFFFFFFFFFFFFFULL; /* constant salt */
  const uint32_t P = slices.size();
  vector<std::thread> pool;
  pool.reserve(P);

  for (uint32_t t = 0; t < P; ++t) {
    pool.emplace_back([&, t] {
      PRF_AES128_fast prf(secret);
      const Slice sl = slices[t];
      uint32_t i = sl.begin;
      const uint32_t end = sl.end;

      for (; i + 4 <= end; i += 4) {
        if (i + 32 < end) {
          _mm_prefetch(reinterpret_cast<const char *>(&table.tuples[i + 32]),
                       _MM_HINT_T0);
        }

        uint32_t keys[4] = {table.tuples[i].key, table.tuples[i + 1].key,
                            table.tuples[i + 2].key, table.tuples[i + 3].key};
        uint64_t salts[4] = {i, i + 1, i + 2, i + 3};
        uint64_t prfOut[4];
        prf(keys, salts, prfOut);
        uint32_t d0 = static_cast<uint32_t>(prfOut[0]);
        uint32_t d1 = static_cast<uint32_t>(prfOut[1]);
        uint32_t d2 = static_cast<uint32_t>(prfOut[2]);
        uint32_t d3 = static_cast<uint32_t>(prfOut[3]);

        uint32_t m0 = -(table.tuples[i].cntSelf == 0);
        uint32_t m1 = -(table.tuples[i + 1].cntSelf == 0);
        uint32_t m2 = -(table.tuples[i + 2].cntSelf == 0);
        uint32_t m3 = -(table.tuples[i + 3].cntSelf == 0);

        table.tuples[i].idx = i;
        table.tuples[i + 1].idx = i + 1;
        table.tuples[i + 2].idx = i + 2;
        table.tuples[i + 3].idx = i + 3;

        table.tuples[i].hashKey =
            prf((table.tuples[i].key & ~m0) | (d0 & m0), kHashSalt);
        table.tuples[i + 1].hashKey =
            prf((table.tuples[i + 1].key & ~m1) | (d1 & m1), kHashSalt);
        table.tuples[i + 2].hashKey =
            prf((table.tuples[i + 2].key & ~m2) | (d2 & m2), kHashSalt);
        table.tuples[i + 3].hashKey =
            prf((table.tuples[i + 3].key & ~m3) | (d3 & m3), kHashSalt);
      }

      for (; i < end; ++i) {
        table.tuples[i].idx = i;
        uint32_t dummy = prf(table.tuples[i].key, i);
        uint32_t mask = -(table.tuples[i].cntSelf == 0);
        table.tuples[i].hashKey =
            prf((table.tuples[i].key & ~mask) | (dummy & mask), kHashSalt);
      }
    });
  }
  for (auto &th : pool)
    th.join();
}
