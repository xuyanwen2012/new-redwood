#pragma once

#include "../PointCloud.hpp"

struct MyFunctor {
  static auto rsqrtf(const float x) { return 1.0f / sqrtf(x); }

  // GPU version
  _REDWOOD_KERNEL Point3F operator()(const Point4F p, const Point3F q) const {
    // For SYCL backend, use negative points to indicate invalid
    if (p.data[0] < 0.0f) return Point3F();

    const auto dx = p.data[0] - q.data[0];
    const auto dy = p.data[1] - q.data[1];
    const auto dz = p.data[2] - q.data[2];
    const auto dist_sqr = dx * dx + dy * dy + dz * dz + 1e-9f;
    const auto inv_dist = rsqrtf(dist_sqr);
    const auto inv_dist3 = inv_dist * inv_dist * inv_dist;
    const auto with_mass = inv_dist3 * p.data[3];
    return {dx * with_mass, dy * with_mass, dz * with_mass};
  }

  // FPGA version
  _REDWOOD_KERNEL Point3D operator()(const Point4D p, const Point3D q) const {
    const auto dx = p.data[0] - q.data[0];
    const auto dy = p.data[1] - q.data[1];
    const auto dz = p.data[2] - q.data[2];
    const auto dist_sqr = dx * dx + dy * dy + dz * dz + 1e-9;
    const auto inv_dist = 1.0 / sqrt(dist_sqr);
    const auto inv_dist3 = inv_dist * inv_dist * inv_dist;
    const auto with_mass = inv_dist3 * p.data[3];
    return {dx * with_mass, dy * with_mass, dz * with_mass};
  }
};