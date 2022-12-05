#pragma once

#include <cmath>

#include "../PointCloud.hpp"

struct MyFunctor {
  // GPU version
  _REDWOOD_KERNEL float operator()(const Point2F p, const Point2F q) const {
    auto lat1 = p.data[0];
    auto lat2 = q.data[0];
    const auto lon1 = p.data[1];
    const auto lon2 = q.data[1];

    const auto dLat = (lat2 - lat1) * M_PI / 180.0f;
    const auto dLon = (lon2 - lon1) * M_PI / 180.0f;

    // convert to radians
    lat1 = lat1 * M_PI / 180.0f;
    lat2 = lat2 * M_PI / 180.0f;

    // apply formula
    float a = powf(sinf(dLat / 2), 2) +
              powf(sinf(dLon / 2), 2) * cosf(lat1) * cosf(lat2);
    constexpr float rad = 6371;
    float c = 2 * asinf(sqrtf(a));
    return rad * c;
  } /*  */

  // GPU version
  _REDWOOD_KERNEL double operator()(const Point2D p, const Point2D q) const {
    auto lat2 = q.data[0];
    const auto lon2 = q.data[1];

    auto lat1 = p.data[0];
    const auto lon1 = p.data[1];

    const auto dLat = (lat2 - lat1) * M_PI / 180.0;
    const auto dLon = (lon2 - lon1) * M_PI / 180.0;

    // convert to radians
    lat1 = lat1 * M_PI / 180.0;
    lat2 = lat2 * M_PI / 180.0;

    // apply formula
    float a =
        pow(sin(dLat / 2), 2) + pow(sin(dLon / 2), 2) * cos(lat1) * cos(lat2);
    constexpr float rad = 6371;
    float c = 2 * asin(sqrt(a));
    return rad * c;
  } /*  */
};
