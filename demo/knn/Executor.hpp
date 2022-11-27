#pragma once

#include <algorithm>
#include <cassert>
#include <limits>
#include <memory>
#include <vector>

#include "../../src/Redwood.hpp"
#include "../nn/KDTree.hpp"
#include "../nn/Kernel.hpp"
#include "KnnSet.hpp"

namespace redwood {
struct Task {
  int query_idx;
  Point4F query_point;
};

namespace dev {
enum class ExecutionState { kWorking, kFinished };

constexpr auto kLogLevel = 1;

// Global reference to the single KD tree
inline std::shared_ptr<kdt::KdTree> tree_ref;
inline std::shared_ptr<std::vector<float>> result_ref;

struct ExecutorStats {
  int leaf_node_reduced = 0;
  int branch_node_reduced = 0;
};

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

class KnnExecutor {
 public:
  KnnExecutor() : task_(), state_(ExecutionState::kFinished), cur_(nullptr) {
    stack_.reserve(16);
  }

  void StartQuery(const Task& task) {
    task_ = task;
    stack_.clear();
    cur_ = nullptr;
    GetReductionResult(0, task.query_idx, &cached_result_set_);
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

        cached_result_set_->Insert(dist);

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
            diff * diff < cached_result_set_->WorstDist()) {
          cur_ = last_cur->GetChild(FlipDir(dir));
        }
      }
    }

    // Done traversals, Write back to final results
    state_ = ExecutionState::kFinished;
  }

  // long tid_;

  // Actually essential data in a executor
  Task task_;
  std::vector<CallStackField> stack_;
  ExecutionState state_;
  kdt::Node* cur_;

  KnnSet<float, 32>*
      cached_result_set_;  // a pointer to the KNN set of 32 float (idx * 32)
};

// -------------------------------------------------------------------------------------------------
// KNN here
// -------------------------------------------------------------------------------------------------

template <typename T>
class KnnExecutorManager {
 public:
  KnnExecutorManager() = delete;

  KnnExecutorManager(const std::shared_ptr<kdt::KdTree> tree,
                     std::vector<Task>& tasks, const int num_batches,
                     const int tid = 0)
      : my_tasks_(tasks), num_batches_(num_batches) {
    // Save reference to
    if (!tree_ref) {
      std::cout << "[DEBUG] kdt::KdTree Reference Set!" << std::endl;
      tree_ref = tree;
    }

    assert(tasks.size() % num_batches == 0);

    // Need to do double buffering
    executors_.resize(2 * num_batches);

    std::cout << "Manager "
              << ":\n"
              << "\tnum queries: " << my_tasks_.size() << '\n'
              << "\tnum batches: " << num_batches_ << '\n'
              << "\tnum executors: " << executors_.size() << '\n'
              << std::endl;
  }

  void StartTraversals() {
    while (!executors_.empty()) {
      auto mid_point = executors_.begin() + executors_.size() / 2;
      for (auto it = executors_.begin(); it != mid_point;) {
        if (!it->Finished()) {
          it->Resume();
          ++it;
          continue;
        }

        if (my_tasks_.empty()) {
          it = executors_.erase(it);
          --mid_point;
        } else {
          const auto task = my_tasks_.back();
          my_tasks_.pop_back();
          it->StartQuery(task);
          ++it;
        }
      }

      ExecuteBatchedKernelsAsync(0,
                                 std::distance(executors_.begin(), mid_point));

      for (auto it = mid_point; it != executors_.end();) {
        if (!it->Finished()) {
          it->Resume();
          ++it;
          continue;
        }

        if (my_tasks_.empty()) {
          it = executors_.erase(it);
        } else {
          const auto task = my_tasks_.back();
          my_tasks_.pop_back();
          it->StartQuery(task);
          ++it;
        }
      }

      ExecuteBatchedKernelsAsync(0, std::distance(mid_point, executors_.end()));
    }
  }

  _NODISCARD ExecutorStats GetStats() const { return stats_; }

 private:
  std::vector<Task>& my_tasks_;
  std::vector<KnnExecutor> executors_;

  const int num_batches_;
  ExecutorStats stats_;
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

    cpu_results_ =
        static_cast<float*>(malloc(sizeof(float) * 32 * my_tasks_.size()));
    std::fill_n(cpu_results_, 32 * my_tasks_.size(),
                std::numeric_limits<float>::max());

    for (int i = 0; i < 4; ++i) {
      std::cout << my_tasks_[i].query_point << std::endl;
    }

    std::cout << "Sequential Manager "
              << ":\n"
              << "\tnum queries: " << my_tasks_.size() << '\n'
              << std::endl;
  }

  void StartTraversals() {
    while (!my_tasks_.empty()) {
      const auto task = my_tasks_.back();
      my_tasks_.pop_back();

      KnnSearchRecursive(tree_ref->GetRoot(), task);
    }
  }

  float* GetCpuResult(const int query_idx) const {
    return cpu_sets_[query_idx].rank;
  }

 protected:
  void KnnSearchRecursive(const kdt::Node* cur, const Task task) {
    static auto kernel_func = MyFunctor();

    if (cur->IsLeaf()) {
      ++stats_.leaf_node_reduced;

      // **** Reduction at leaf node ****
      const auto leaf_size = tree_ref->params_.leaf_max_size;
      for (int i = 0; i < leaf_size; ++i) {
        const auto p =
            tree_ref->GetNodeContentTable()[cur->uid * leaf_size + i];
        const auto dist = kernel_func(p, task.query_point);

        cpu_sets_[task.query_idx].Insert(dist);
      }
      // **********************************
    } else {
      const unsigned accessor_idx =
          tree_ref->v_acc_[cur->node_type.tree.idx_mid];

      ++stats_.branch_node_reduced;

      // **** Reduction at branch node ****
      const auto dist = kernel_func(
          tree_ref->GetNodeContentTable()[accessor_idx], task.query_point);
      cpu_sets_[task.query_idx].Insert(dist);

      // **********************************

      const auto axis = cur->node_type.tree.axis;
      const auto train = tree_ref->data_set_[accessor_idx].data[axis];
      const auto dir = task.query_point.data[axis] < train ? kdt::Dir::kLeft
                                                           : kdt::Dir::kRight;

      KnnSearchRecursive(cur->GetChild(dir), task);

      if (const auto diff = task.query_point.data[axis] - train;
          diff * diff < cpu_sets_[task.query_idx].WorstDist()) {
        KnnSearchRecursive(cur->GetChild(FlipDir(dir)), task);
      }
    }
  }

 private:
  std::vector<Task>& my_tasks_;
  ExecutorStats stats_;

  union {
    KnnSet<float, 32>* cpu_sets_;
    float* cpu_results_;
  };
};
}  // namespace dev
}  // namespace redwood
