#pragma once
#include "external_memory/algorithm/kway_butterfly_sort.hpp"
#include "external_memory/stdvector.hpp"
#include "external_memory/virtualvector.hpp"

template <const uint64_t dim>
struct Point {
  double data[dim];
  double distance2(const Point<dim>& other) const {
    double dist2 = 0;
    for (uint64_t i = 0; i < dim; i++) {
      double diff = data[i] - other.data[i];
      dist2 += diff * diff;
    }
    return dist2;
  }
  double distance(const Point<dim>& other) const {
    return sqrt(distance2(other));
  }
};
