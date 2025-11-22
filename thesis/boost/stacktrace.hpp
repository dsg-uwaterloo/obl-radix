#pragma once
#include <ostream>

namespace boost {
namespace stacktrace {
// Lightweight stub to avoid pulling Boost when tracing is not needed.
struct stacktrace {
  stacktrace() = default;
};

inline std::ostream &operator<<(std::ostream &os, const stacktrace &) {
  return os;
}
} // namespace stacktrace
} // namespace boost
