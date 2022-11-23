#pragma once

#include "../PointCloud.hpp"

struct MyFunctor {
  // GPU version
  _REDWOOD_KERNEL float operator()(const Point4F p, const Point4F q) const {
    auto dist = float();

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const auto diff = p.data[i] - q.data[i];
      dist += diff * diff;
    }

    return sqrtf(dist);
  }
};
