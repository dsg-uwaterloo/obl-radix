#ifndef INPUTS_H
#define INPUTS_H

#include <cctype>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#define DATA_LENGTH 12

struct Record {
  uint32_t key;
  uint32_t cntSelf;
  uint32_t cntExpand;
  uint32_t hashKey;
  uint32_t idx;
  char pay[DATA_LENGTH];

  bool operator<(Record const &o) const { return key < o.key; }
} __attribute__((aligned(32)));

inline void maskedCopyRecord32(const Record *src, Record *dst,
                               std::uint64_t mask) // 0 or ~0
{
#if defined(__AVX2__)
  __m256i vSrc = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src));
  __m256i vDst = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(dst));
  __m256i m =
      _mm256_set1_epi64x(static_cast<long long>(mask)); // 256-bit broadcast
  __m256i res =
      _mm256_or_si256(_mm256_and_si256(vSrc, m), _mm256_andnot_si256(m, vDst));
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), res);
#else // portable SSE2 fallback
  std::cout << "SSE2" << std::endl;
  __m128i srcLo = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src));
  __m128i srcHi = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src) + 1);
  __m128i dstLo = _mm_loadu_si128(reinterpret_cast<const __m128i *>(dst));
  __m128i dstHi = _mm_loadu_si128(reinterpret_cast<const __m128i *>(dst) + 1);
  __m128i m = _mm_set1_epi32(static_cast<int>(mask));
  __m128i outLo =
      _mm_or_si128(_mm_and_si128(srcLo, m), _mm_andnot_si128(m, dstLo));
  __m128i outHi =
      _mm_or_si128(_mm_and_si128(srcHi, m), _mm_andnot_si128(m, dstHi));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), outLo);
  _mm_storeu_si128(reinterpret_cast<__m128i *>(dst) + 1, outHi);
#endif
}

// returns true if 's' is empty or contains only whitespace chars
static bool is_blank_line(const std::string &s) {
  for (char c : s) {
    if (!std::isspace(static_cast<unsigned char>(c))) {
      return false;
    }
  }
  return true;
}

// Reads two tables from a file whose first non-empty line is: n0 n1
// (any number of blank lines are skipped), then exactly n0 + n1 data lines:
// <key> <rest-of-line> where the "rest-of-line" (including spaces) goes into
// .paySelf.
inline bool load_two_tables(const std::string &input_path,
                            std::vector<Record> &table0,
                            std::vector<Record> &table1) {
  std::ifstream in(input_path);
  if (!in) {
    std::cerr << "Error: cannot open \"" << input_path << "\"\n";
    return false;
  }

  // --- 1) read header, skipping any blank lines ---
  size_t n0 = 0, n1 = 0;
  {
    std::string header_line;
    while (std::getline(in, header_line)) {
      if (is_blank_line(header_line))
        continue;
      std::istringstream hdr(header_line);
      if (!(hdr >> n0 >> n1)) {
        std::cerr << "Error: malformed header: \"" << header_line << "\"\n";
        return false;
      }
      break;
    }
  }

  table0.clear();
  table0.reserve(n0);
  table1.clear();
  table1.reserve(n1);

  auto read_n = [&](size_t count, auto &tbl) -> bool {
    size_t read = 0;
    std::string line;
    while (read < count && std::getline(in, line)) {
      if (is_blank_line(line))
        continue;

      std::istringstream iss(line);
      Record rec;

      // parse the key
      std::uint32_t temp_key;
      if (!(iss >> temp_key)) {
        std::cerr << "Error parsing key in line: \"" << line << "\"\n";
        return false;
      }
      rec.key = temp_key;

      // the rest of the line—including spaces—is the payload
      std::string rest;
      std::getline(iss, rest);
      // strip a single leading space if present
      if (!rest.empty() && rest.front() == ' ')
        rest.erase(0, 1);

      // Copy payload into char array, ensuring null termination
      std::memset(rec.pay, 0, DATA_LENGTH);
      size_t copy_len =
          std::min(rest.length(), static_cast<size_t>(DATA_LENGTH - 1));
      std::strncpy(rec.pay, rest.c_str(), copy_len);
      rec.pay[copy_len] = '\0';

      // Initialize other fields
      rec.idx = read;
      rec.cntSelf = 0;
      rec.cntExpand = 0;

      tbl.push_back(rec);
      ++read;
    }
    if (read < count) {
      std::cerr << "Error: only read " << read << " of " << count
                << " requested records\n";
      return false;
    }
    return true;
  };

  if (!read_n(n0, table0))
    return false;
  if (!read_n(n1, table1))
    return false;

  return true;
}

#endif // INPUTS_H
