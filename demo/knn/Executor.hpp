#pragma once

#include <cassert>
#include <cstdlib>
#include <memory>
#include <vector>

#include "../../src/Redwood.hpp"
#include "KDTree.hpp"
#include "Kernel.cuh"

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

namespace dev {

struct ExecutorStats {
  int leaf_node_reduced = 0;
  int branch_node_reduced = 0;
};

// Pure CPU Sequential Executor
template <typename T>
class KnnCpuManager {
  using QueryPointT = Point<4, T>;

 public:
  KnnCpuManager(std::shared_ptr<kdt::KdTree> tree,
                const QueryPointT* my_query_points, const int my_m,
                const int tid = 0)
      : tid_(tid), tree_(tree), my_query_points_(my_query_points), my_m_(my_m) {
    tasks_todo_.resize(my_m);

    std::cout << "CpuManager " << tid_ << ":\n"
              << "\tmy_m: " << my_m_ << '\n'
              << std::endl;

    const auto k = 32;
    my_results.resize(my_m * k);
    my_k_sets.reserve(my_m);
    for (int i = 0; i < my_m; ++i) {
      my_k_sets.emplace_back(k);
      my_k_sets[i].Init(my_results.data() + i * k);
    }

    std::iota(tasks_todo_.begin(), tasks_todo_.end(), 0u);
    std::reverse(tasks_todo_.begin(), tasks_todo_.end());
  }

  void StartTraversals() {
    // Simple Sequential Traversal for CPU backend
    while (!tasks_todo_.empty()) {
      auto q_idx = tasks_todo_.back();
      tasks_todo_.pop_back();

      // redwood::StartQuery(tid_, q_idx);
      KnnSearchRecursive(tree_->GetRoot(), q_idx);
    }
  }

  _NODISCARD const T* GetCpuResult(const int query_idx, const int k) const {
    return &my_results[query_idx * k];
  }

  _NODISCARD ExecutorStats GetStats() const { return stats_; }

 protected:
  void KnnSearchRecursive(const kdt::Node* cur, const unsigned query_idx) {
    static auto kernel_func = MyFunctor();

    if (cur->IsLeaf()) {
      const auto q = my_query_points_[query_idx];

      ++stats_.leaf_node_reduced;

      // **** Reduction at leaf node ****
      const auto leaf_size = tree_->params_.leaf_max_size;
      for (int i = 0; i < leaf_size; ++i) {
        const auto p = tree_->GetNodeContentTable()[cur->uid * leaf_size + i];
        const auto dist = kernel_func(p, q);

        my_k_sets[query_idx].AddPoint(dist);
        // cpu_results[query_idx].Insert(dist);
      }
      // **********************************

    } else {
      const unsigned accessor_idx = tree_->v_acc_[cur->node_type.tree.idx_mid];

      ++stats_.branch_node_reduced;

      // **** Reduction at branch node ****
      const auto dist = kernel_func(tree_->GetNodeContentTable()[accessor_idx],
                                    my_query_points_[query_idx]);
      my_k_sets[query_idx].AddPoint(dist);

      // **********************************

      const auto axis = cur->node_type.tree.axis;
      const auto train = tree_->data_set_[accessor_idx].data[axis];
      const auto dir = my_query_points_[query_idx].data[axis] < train
                           ? kdt::Dir::kLeft
                           : kdt::Dir::kRight;

      KnnSearchRecursive(cur->GetChild(dir), query_idx);

      const auto diff = my_query_points_[query_idx].data[axis] - train;
      if (diff * diff < my_k_sets[query_idx].WorstDist()) {
        KnnSearchRecursive(cur->GetChild(FlipDir(dir)), query_idx);
      }
    }
  }

 private:
  // Associated to a CUDA Stream or a SYCL queue
  int tid_;
  std::shared_ptr<kdt::KdTree> tree_;

  const Point4F* my_query_points_;

  // A list of local 'query_idx', with respect to global query_idx
  std::vector<unsigned> tasks_todo_;
  const int my_m_;

  ExecutorStats stats_;

  // Keep results here
  std::vector<T> my_results;
  std::vector<KnnResultSet<T>> my_k_sets;
};

}  // namespace dev

}  // namespace redwood