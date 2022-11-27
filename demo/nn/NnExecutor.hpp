#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include "../../src/Redwood.hpp"
#include "ExecutorManager.hpp"
#include "KDTree.hpp"
#include "Kernel.hpp"

namespace redwood {

namespace dev {

// -------------------------------------------------------------------------------------------------
// SYCL Executor related here
// -------------------------------------------------------------------------------------------------

#ifndef REDWOOD_IN_CPU

// Basically, a pointer, an int32, a float, an int
struct CallStackField {
  kdt::Node* current;
  int axis;
  float train;
  kdt::Dir dir;
};

class NnExecutor {
 public:
  NnExecutor() : task_(), state_(ExecutionState::kFinished), cur_(nullptr) {
    stack_.reserve(16);
  }

  void StartQuery(const Task& task) {
    task_ = task;
    stack_.clear();
    cur_ = nullptr;
    GetReductionResult(0, task.query_idx, &cached_result_addr_);
    Execute();
  }

  void Resume() { Execute(); }

  _NODISCARD bool Finished() const {
    return state_ == ExecutionState::kFinished;
  }

 private:
  void Execute() {
    if (state_ == ExecutionState::kWorking) goto my_resume_point;

    state_ = ExecutionState::kWorking;
    cur_ = tree_ref->root_;

    // Begin Iteration
    while (cur_ != nullptr || !stack_.empty()) {
      // Traverse all the way to left most leaf node
      while (cur_ != nullptr) {
        if (cur_->IsLeaf()) {
          ReduceLeafNodeWithTask(0, cur_->uid, &task_);

          // **** Coroutine Reuturn ****
          return;
        my_resume_point:
          // ****************************

          cur_ = nullptr;
          continue;
        }

        // **** Reduction at tree node ****

        const unsigned accessor_idx =
            tree_ref->v_acc_[cur_->node_type.tree.idx_mid];

        auto kernel_func = MyFunctor();
        const float dist =
            kernel_func(tree_ref->data_set_[accessor_idx], task_.query_point);

        *cached_result_addr_ = std::min(*cached_result_addr_, dist);

        // **********************************

        const int axis = cur_->node_type.tree.axis;
        const float train = tree_ref->data_set_[accessor_idx].data[axis];
        const kdt::Dir dir = task_.query_point.data[axis] < train
                                 ? kdt::Dir::kLeft
                                 : kdt::Dir::kRight;

        stack_.push_back({cur_, axis, train, dir});
        cur_ = cur_->GetChild(dir);
      }

      // We resume back from Break point, and now we are still in the branch
      // node, we can check if there's any thing left on the stack.
      if (!stack_.empty()) {
        const auto [last_cur, axis, train, dir] = stack_.back();
        stack_.pop_back();

        // Check if there is a possibility of the NN lies on the other half
        // If the difference between the query point and the other splitting
        // plane is greater than the current found minimum distance, then it is
        // impossible to have a NN there.
        if (const auto diff = task_.query_point.data[axis] - train;
            diff * diff < *cached_result_addr_) {
          cur_ = last_cur->GetChild(FlipDir(dir));
        }
      }
    }

    // Done traversals, Write back to final results
    state_ = ExecutionState::kFinished;
  }

  // int tid_;

  // Actually essential data in a executor
  Task task_;
  std::vector<CallStackField> stack_;
  ExecutionState state_;
  kdt::Node* cur_;

  float* cached_result_addr_;  // a pointer to the USM of 1 float
};

#endif

// -------------------------------------------------------------------------------------------------
// CPU Sequential here
// -------------------------------------------------------------------------------------------------

template <typename T>
class SequentialManager {
 public:
  SequentialManager() = delete;

  SequentialManager(const std::shared_ptr<kdt::KdTree> tree,
                    std::vector<Task>& tasks, const int tid = 0)
      : my_tasks_(tasks) {
    // Save reference to
    if (!tree_ref) {
      std::cout << "[DEBUG] kdt::KdTree Reference Set!" << std::endl;
      tree_ref = tree;
    }

    result_.resize(my_tasks_.size());
    std::fill(result_.begin(), result_.end(),
              std::numeric_limits<float>::max());

    std::cout << "Sequential Manager "
              << ":\n"
              << "\tnum queries: " << my_tasks_.size() << '\n'
              << std::endl;
  }

  void StartTraversals() {
    while (!my_tasks_.empty()) {
      const auto task = my_tasks_.back();
      my_tasks_.pop_back();

      NnSearchRecursive(tree_ref->GetRoot(), task);
    }
  }

  float GetCpuResult(const int query_idx) const { return result_[query_idx]; }

 protected:
  void NnSearchRecursive(const kdt::Node* cur, const Task task) {
    static auto kernel_func = MyFunctor();

    if (cur->IsLeaf()) {
      // ++stats_.leaf_node_reduced;

      // **** Reduction at leaf node ****
      const auto leaf_size = tree_ref->params_.leaf_max_size;
      for (int i = 0; i < leaf_size; ++i) {
        const auto p =
            tree_ref->GetNodeContentTable()[cur->uid * leaf_size + i];
        const auto dist = kernel_func(p, task.query_point);

        // cpu_sets_[task.query_idx].Insert(dist);
        result_[task.query_idx] = std::min(result_[task.query_idx], dist);
      }
      // **********************************
    } else {
      const unsigned accessor_idx =
          tree_ref->v_acc_[cur->node_type.tree.idx_mid];

      // ++stats_.branch_node_reduced;

      // **** Reduction at branch node ****
      const auto dist = kernel_func(
          tree_ref->GetNodeContentTable()[accessor_idx], task.query_point);
      // cpu_sets_[task.query_idx].Insert(dist);
      result_[task.query_idx] = std::min(result_[task.query_idx], dist);

      // **********************************

      const auto axis = cur->node_type.tree.axis;
      const auto train = tree_ref->data_set_[accessor_idx].data[axis];
      const auto dir = task.query_point.data[axis] < train ? kdt::Dir::kLeft
                                                           : kdt::Dir::kRight;

      NnSearchRecursive(cur->GetChild(dir), task);

      if (const auto diff = task.query_point.data[axis] - train;
          diff * diff < result_[task.query_idx]) {
        NnSearchRecursive(cur->GetChild(FlipDir(dir)), task);
      }
    }
  }

 private:
  std::vector<Task>& my_tasks_;

  std::vector<float> result_;
};
}  // namespace dev
}  // namespace redwood
