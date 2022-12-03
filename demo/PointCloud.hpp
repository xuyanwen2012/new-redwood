#pragma once

#include <algorithm>
#include <iostream>

#include "../src/Utils.hpp"

template <int Dim, typename T>
struct Point {
  Point() = default;

  static constexpr auto dim = Dim;

  T data[Dim];

  _REDWOOD_KERNEL Point<Dim, T> operator/(const T a) const {
    Point<Dim, T> result;
    for (int i = 0; i < Dim; ++i) {
      result.data[i] = data[i] / a;
    }
    return result;
  }

  _REDWOOD_KERNEL Point<Dim, T> operator*(const T a) const {
    Point<Dim, T> result;
    for (int i = 0; i < Dim; ++i) {
      result.data[i] = data[i] * a;
    }
    return result;
  }

  _REDWOOD_KERNEL Point<Dim, T> operator+(const Point<Dim, T>& pos) const {
    Point<Dim, T> result;
    for (int i = 0; i < Dim; ++i) {
      result.data[i] = data[i] + pos.data[i];
    }
    return result;
  }

  _REDWOOD_KERNEL Point<Dim, T> operator-(const Point<Dim, T>& pos) const {
    Point<Dim, T> result;
    for (int i = 0; i < Dim; ++i) {
      result.data[i] = data[i] - pos.data[i];
    }
    return result;
  }

  _REDWOOD_KERNEL Point<Dim, T>& operator+=(const Point<Dim, T>& rhs) {
    for (int i = 0; i < Dim; ++i) {
      this->data[i] += rhs.data[i];
    }
    return *this;
  }

  _REDWOOD_KERNEL Point<Dim, T>& operator++() {
    ++data[0];
    return *this;
  };

  // _REDWOOD_KERNEL void operator=(const Point<Dim, T>& rhs) {
  //   for (int i = 0; i < Dim; ++i) {
  //     data[i] = rhs.data[i];
  //   }
  // }
};

template <int Dim, typename T>
Point<Dim, T> MakeRandomPoint() {
  Point<Dim, T> pt;
  for (int i = 0; i < Dim; ++i) {
    pt.data[i] = my_rand<T>();
  }
  return pt;
}

template <int Dim, typename T>
_REDWOOD_KERNEL bool operator==(const Point<Dim, T>& a,
                                const Point<Dim, T>& b) {
  for (int i = 0; i < Dim; ++i) {
    if (a.data[i] != b.data[i]) return false;
  }
  return true;
}

template <int Dim, typename T>
_REDWOOD_KERNEL bool operator!=(const Point<Dim, T>& a,
                                const Point<Dim, T>& b) {
  return !(a == b);
}

template <int Dim, typename T>
std::ostream& operator<<(std::ostream& os, const Point<Dim, T>& dt) {
  os << '(';
  for (int i = 0; i < Dim - 1; ++i) {
    os << dt.data[i] << ", ";
  }
  os << dt.data[Dim - 1] << ')';
  return os;
}

using Point2F = Point<2, float>;
using Point2D = Point<2, double>;
using Point3F = Point<3, float>;
using Point4F = Point<4, float>;
using Point4D = Point<4, double>;
using Point3D = Point<3, double>;