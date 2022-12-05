#pragma once

#include <cmath>

#include "../PointCloud.hpp"

namespace kernel {
static auto rsqrtf(const float x) { return 1.0f / sqrtf(x); }

constexpr auto kKdeConstant = 0.2f;

// "4F -> 3F -> 3F" functors
struct GravityFunctor;
struct GaussianFunctor;
struct TopHatFunctor;

using MyFunctor = GravityFunctor;

struct GravityFunctor {
  // GPU version
  _REDWOOD_KERNEL Point3F operator()(const Point4F p, const Point3F q) const {
    const auto dx = p.data[0] - q.data[0];
    const auto dy = p.data[1] - q.data[1];
    const auto dz = p.data[2] - q.data[2];
    const auto dist_sqr = dx * dx + dy * dy + dz * dz + 1e-9f;
    const auto inv_dist = rsqrtf(dist_sqr);
    const auto inv_dist3 = inv_dist * inv_dist * inv_dist;
    const auto with_mass = inv_dist3 * p.data[3];
    return {dx * with_mass, dy * with_mass, dz * with_mass};
  }
};

struct GaussianFunctor {
  // GPU version
  _REDWOOD_KERNEL Point3F operator()(const Point4F p, const Point3F q) const {
    constexpr auto h = kKdeConstant;
    const auto dx = p.data[0] - q.data[0];
    const auto dy = p.data[1] - q.data[1];
    const auto dz = p.data[2] - q.data[2];
    const auto dist_sqr = dx * dx + dy * dy + dz * dz + 1e-9f;
    const auto dist = sqrtf(dist_sqr);
    const auto exp = std::exp(-(dist * dist / (2 * h * h)));
    const auto with_mass = exp * p.data[3];
    return {dx * with_mass, dy * with_mass, dz * with_mass};
  }
};

struct TopHatFunctor {
  // GPU version
  _REDWOOD_KERNEL Point3F operator()(const Point4F p, const Point3F q) const {
    constexpr auto h = 5.0f;
    const auto dx = p.data[0] - q.data[0];
    const auto dy = p.data[1] - q.data[1];
    const auto dz = p.data[2] - q.data[2];
    const auto dist_sqr = dx * dx + dy * dy + dz * dz + 1e-9f;
    const auto dist = sqrtf(dist_sqr);
    const auto a = dist < h ? 1.0f : 0.0f;
    const auto with_mass = a * p.data[3];
    return {dx * with_mass, dy * with_mass, dz * with_mass};
  }
};

}  // namespace kernel
