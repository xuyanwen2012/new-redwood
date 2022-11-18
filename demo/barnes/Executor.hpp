#pragma once

#include <cassert>

#include "../../src/Redwood.hpp"
#include "Octree.hpp"

namespace redwood {

namespace dev {

struct ExecutorStats {
  int leaf_node_reduced = 0;
  int branch_node_reduced = 0;
};

// Data parallel executor.
template <typename T>
class BarnesExecutorManager {
  using QueryPointT = Point<3, T>;

 public:
  BarnesExecutorManager(const oct::Octree<T>& tree,
                        const QueryPointT* my_query_points, const int my_m,
                        const int num_batches, const int tid = 0)
      : tid_(tid),
        tree_(tree),
        my_query_points_(my_query_points),
        my_m_(my_m),
        num_batches_(num_batches) {
    assert(my_m % num_batches == 0);
    tasks_todo_.resize(my_m);

    std::iota(tasks_todo_.begin(), tasks_todo_.end(), 0u);
    std::reverse(tasks_todo_.begin(), tasks_todo_.end());
  }

  void StartTraversal() {
    const auto iterations = my_m_ / num_batches_;

    for (int j = 0; j < iterations; ++j) {
      for (int i = 0; i < num_batches_; ++i) {
        cur_query_index_ = tasks_todo_.back();
        tasks_todo_.pop_back();

        redwood::StartQuery(tid_, cur_query_index_);

        ComputeForceRecursive(tree_.GetRoot());
      }
      redwood::ExecuteBatchedKernelsAsync(tid_);
    }
  }

  _NODISCARD ExecutorStats GetStats() const { return stats_; }

 protected:
  static T ComputeThetaValue(const oct::Node<T>* node, const QueryPointT pos) {
    const auto com = node->CenterOfMass();
    auto norm_sqr = T(1e-9);

    for (int i = 0; i < 3; ++i) {
      const auto diff = com->data[i] - pos.data[i];
      norm_sqr += diff * diff;
    }

    const auto norm = sqrt(norm_sqr);
    return node->bounding_box.dimension.data[0] / norm;
  }

  void ComputeForceRecursive(const oct::Node<T>* cur) {
    if (cur->IsLeaf()) {
      if (cur->bodies.empty()) return;

      ++stats_.leaf_node_reduced;

      // std::cout << cur_query_index_ << ": Leaf" << cur->uid << std::endl;

      redwood::ReduceLeafNode(tid_, cur->uid, cur_query_index_);
    } else {
      const auto my_theta =
          ComputeThetaValue(cur, my_query_points_[cur_query_index_]);
      if (my_theta < tree_.GetParams().theta) {
        ++stats_.branch_node_reduced;
        // std::cout << cur_query_index_ << ": Branch" << *cur->CenterOfMass()
        //           << "\t ComputeThetaValue:" << my_theta << std::endl;

        redwood::ReduceBranchNode(tid_, cur->CenterOfMass(), cur_query_index_);
      } else
        for (const auto child : cur->children)
          if (child != nullptr) ComputeForceRecursive(child);
    }
  }

 private:
  // Associated to a CUDA Stream or a SYCL queue
  int tid_;
  const oct::Octree<T>& tree_;
  const Point3F* my_query_points_;
  unsigned cur_query_index_;

  // A list of local 'query_idx', with respect to global query_idx
  std::vector<unsigned> tasks_todo_;
  const int my_m_;
  const int num_batches_;

  ExecutorStats stats_;
  // Keep results here
};

}  // namespace dev

}  // namespace redwood