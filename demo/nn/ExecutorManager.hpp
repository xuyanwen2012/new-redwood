#pragma once
#include <algorithm>
#include <cassert>
#include <memory>
#include <vector>

#include "KDTree.hpp"

namespace redwood {
struct Task {
  int query_idx;
  Point4F query_point;
};

namespace dev {

// Global reference to the single KD tree
inline std::shared_ptr<kdt::KdTree> tree_ref;

enum class ExecutionState { kWorking, kFinished };

template <typename T, typename ExecutorT>
class ExecutorManager {
 public:
  ExecutorManager() = delete;

  ExecutorManager(const std::shared_ptr<kdt::KdTree> tree,
                  std::vector<Task>& tasks, const int num_batches,
                  const int tid = 0)
      : tid_(tid), my_tasks_(tasks), num_batches_(num_batches) {
    // Save reference to
    if (!tree_ref) {
      std::cout << "[DEBUG] kdt::KdTree Reference Set!" << std::endl;
      tree_ref = tree;
    }

    assert(tasks.size() % num_batches == 0);

    // Need to do double buffering
    executors_.resize(2 * num_batches);
    for (auto& exe : executors_) {
      exe.Init(tid_);
    }

    std::cout << "Manager (" << tid_ << "):\n"
              << "\tnum queries: " << my_tasks_.size() << '\n'
              << "\tnum batches: " << num_batches_ << '\n'
              << "\tnum executors: " << executors_.size() << '\n'
              << std::endl;
  }

  void StartTraversals() {
    std::cout << "Manager (" << tid_ << ") started.\n";

    while (!executors_.empty()) {
      // std::cout << '[' << tid_ << ']' << " tasks left: " << my_tasks_.size()
      //           << std::endl;

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

      ExecuteBatchedKernelsAsync(tid_,
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

      ExecuteBatchedKernelsAsync(tid_,
                                 std::distance(mid_point, executors_.end()));
    }
  }

  //   _NODISCARD ExecutorStats GetStats() const { return stats_; }

 private:
  int tid_;
  std::vector<Task>& my_tasks_;
  std::vector<ExecutorT> executors_;

  const int num_batches_;
  //   ExecutorStats stats_;
};
}  // namespace dev
}  // namespace redwood