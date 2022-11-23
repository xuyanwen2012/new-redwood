#pragma once

#include "../../src/Utils.hpp"

namespace redwood {

template <typename _DistanceType, typename _CountType = size_t>
class KnnResultSet {
 public:
  using DistanceType = _DistanceType;
  using CountType = _CountType;

 private:
  DistanceType* dists;
  CountType capacity;
  CountType count;

 public:
  explicit KnnResultSet(CountType capacity_)
      : dists(nullptr), capacity(capacity_), count(0) {}

  void Init(DistanceType* dists_) {
    dists = dists_;
    count = 0;
    if (capacity)
      dists[capacity - 1] = (std::numeric_limits<DistanceType>::max)();
  }

  _NODISCARD CountType Size() const { return count; }

  _NODISCARD bool Full() const { return count == capacity; }

  bool AddPoint(DistanceType dist) {
    CountType i;
    for (i = count; i > 0; --i) {
      if (dists[i - 1] > dist) {
        if (i < capacity) {
          dists[i] = dists[i - 1];
        }
      } else
        break;
    }
    if (i < capacity) {
      dists[i] = dist;
    }
    if (count < capacity) ++count;

    // tell caller that the search shall continue
    return true;
  }

  _NODISCARD DistanceType WorstDist() const { return dists[capacity - 1]; }
};
}  // namespace redwood