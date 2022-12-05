#pragma once

#include <cassert>
#include <memory>
#include <vector>

#include "../../src/Redwood.hpp"
#include "../PointCloud.hpp"
#include "Kernel.hpp"
#include "Octree.hpp"

namespace redwood {
template <typename QueryT>
struct Task {
  int query_idx;
  QueryT query_point;
};

namespace dev {
constexpr auto kLogLevel = 1;

struct ExecutorStats {
  int leaf_node_reduced = 0;
  int branch_node_reduced = 0;
};

// -------------------------------------------------------------------------------------------------
// Barnes here
// -------------------------------------------------------------------------------------------------

class BarnesExecutorManager {
 public:
  BarnesExecutorManager() = delete;

  BarnesExecutorManager(std::shared_ptr<oct::Octree<float>> tree,
                        std::vector<Task<Point3F>>& tasks,
                        const int num_batches, const int batch_size,
                        const int tid = 0)
      : my_tasks_(tasks),
        my_tasks_copy_(tasks.begin(), tasks.end()),
        tree_ref_(std::move(tree)),
        num_batches_(num_batches),
        batch_size_(batch_size),
        tid_(tid) {
    assert(tasks.size() % num_batches == 0);

    redwood::SetQueryPoints(tid, my_tasks_copy_.data(), my_tasks_.size());

    const auto iterations = my_tasks_.size() / num_batches_;

    std::cout << "Manager "
              << ":\n"
              << "\tnum queries: " << my_tasks_.size() << '\n'
              << "\tnum batches: " << num_batches_ << '\n'
              << "\tbatch size : " << batch_size_ << '\n'
              << "\titerations : " << iterations << '\n'
              << std::endl;
  }

  void StartTraversals() {
    while (!my_tasks_.empty()) {
      const auto task = my_tasks_.back();
      my_tasks_.pop_back();

      StartQuery(tid_, &task);
      ComputeForceRecursive(tree_ref_->GetRoot(), task);
      ExecuteBatchedKernelsAsync(tid_, num_batches_);
    }
  }

  _NODISCARD ExecutorStats GetStats() const { return stats_; }

 protected:
  static float ComputeThetaValue(const oct::Node<float>* node,
                                 const Point3F pos) {
    const auto com = node->CenterOfMass();
    auto norm_sqr = 1e-9f;

    for (int i = 0; i < 3; ++i) {
      const auto diff = com->data[i] - pos.data[i];
      norm_sqr += diff * diff;
    }

    const auto norm = sqrtf(norm_sqr);
    return node->bounding_box.dimension.data[0] / norm;
  }

  void ComputeForceRecursive(const oct::Node<float>* cur,
                             const Task<Point3F>& task) {
    constexpr auto functor = kernel::MyFunctor();
    if (cur->IsLeaf()) {
      if (cur->bodies.empty()) return;

      ++stats_.leaf_node_reduced;

      ReduceLeafNode(tid_, cur->uid, task.query_idx);

    } else {
      if (const auto my_theta = ComputeThetaValue(cur, task.query_point);
          my_theta < tree_ref_->GetParams().theta) {
        ++stats_.branch_node_reduced;

        ReduceBranchNode(tid_, cur->CenterOfMass(), task.query_idx);
      } else {
        for (const auto child : cur->children) {
          if (child != nullptr) ComputeForceRecursive(child, task);
        }
      }
    }
  }

 private:
  std::vector<Task<Point3F>>& my_tasks_;
  const std::vector<Task<Point3F>> my_tasks_copy_;

  const std::shared_ptr<oct::Octree<float>> tree_ref_;
  const int num_batches_;
  const int batch_size_;

  int tid_;
  ExecutorStats stats_;
};
}  // namespace dev
}  // namespace redwood
