#pragma once
#include <cstddef>
#include <cstdint>
#include <iterator>

// Minimal adapter to let oshuffle operate directly on row_t buffers
// without copying into StdVector or NonCachedVector.
template <typename T>
struct RawVector {
  T* data;
  std::size_t n;
  static constexpr std::size_t item_per_page = 1;

  struct Iterator {
    T* p;
    using vector_type = RawVector<T>;
    static constexpr bool random_access = true;
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;
    vector_type* vec = nullptr;

    T& operator*() const { return *p; }
    Iterator& operator++() {
      ++p;
      return *this;
    }
    friend Iterator operator+(const Iterator& it, std::size_t off) {
      return Iterator{it.p + off};
    }
    friend Iterator operator-(const Iterator& it, std::size_t off) {
      return Iterator{it.p - off};
    }
    friend std::size_t operator-(const Iterator& a, const Iterator& b) {
      return static_cast<std::size_t>(a.p - b.p);
    }
    friend bool operator<(const Iterator& a, const Iterator& b) {
      return a.p < b.p;
    }
    friend bool operator<=(const Iterator& a, const Iterator& b) {
      return a.p <= b.p;
    }
    friend bool operator==(const Iterator& a, const Iterator& b) {
      return a.p == b.p;
    }
    friend bool operator!=(const Iterator& a, const Iterator& b) {
      return a.p != b.p;
    }

    vector_type& getVector() { return *vec; }
    static vector_type* getNullVector() { return nullptr; }
  };

  Iterator begin() { return Iterator{data, this}; }
  Iterator end() { return Iterator{data + n, this}; }
  std::size_t size() const { return n; }

  struct PrefetchReader {
    using value_type = T;
    using iterator_type = Iterator;
    Iterator it;
    Iterator end;

    PrefetchReader() : it{nullptr}, end{nullptr} {}
    PrefetchReader(Iterator b, Iterator e, uint32_t = 0) : it(b), end(e) {}

    void init(Iterator b, Iterator e, uint32_t = 0) {
      it = b;
      end = e;
    }
    const T& get() { return *it; }
    const T& read() {
      const T& v = *it;
      ++it;
      return v;
    }
    bool eof() const { return !(it < end); }
    std::size_t size() const { return end - it; }
  };

  struct Writer {
    using value_type = T;
    using iterator_type = Iterator;
    Iterator it;
    Iterator end;

    Writer() : it{nullptr}, end{nullptr} {}
    Writer(Iterator b, Iterator e, uint32_t = 0) : it(b), end(e) {}

    void init(Iterator b, Iterator e, uint32_t = 0) {
      it = b;
      end = e;
    }
    void write(const T& v) {
      *it = v;
      ++it;
    }
    bool eof() const { return !(it < end); }
    std::size_t size() const { return end - it; }
    void flush() {}
  };
};
