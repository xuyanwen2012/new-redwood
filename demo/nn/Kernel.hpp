
#pragma once

#include "../PointCloud.hpp"

namespace kernel {

// "4F -> 3F -> 3F" functors
struct EuclideanFunctor;
struct ManhattanFunctor;
struct ChebyshevFunctor;

using MyFunctor = EuclideanFunctor;

struct EuclideanFunctor {
  // GPU version
  _REDWOOD_KERNEL float operator()(const Point4F p, const Point4F q) const {
    auto dist = float();

    for (int i = 0; i < 4; ++i) {
      const auto diff = p.data[i] - q.data[i];
      dist += diff * diff;
    }

    return sqrtf(dist);
  }
};

struct ManhattanFunctor {
  // GPU version
  _REDWOOD_KERNEL float operator()(const Point4F p, const Point4F q) const {
    auto dist = float();

    for (int i = 0; i < 4; ++i) {
      const auto diff = p.data[i] - q.data[i];
      dist += fabs(diff);
    }

    return dist;
  }
};

struct ChebyshevFunctor {
  // GPU version
  _REDWOOD_KERNEL float operator()(const Point4F p, const Point4F q) const {
    auto dist = float();

    for (int i = 0; i < 4; ++i) {
      const auto diff = fabs(p.data[i] - q.data[i]);
      dist = std::max(dist, static_cast<float>(diff));
    }

    return dist;
  }
};

}  // namespace kernel
