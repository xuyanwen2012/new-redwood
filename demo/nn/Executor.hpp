#pragma once

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <limits>
#include <memory>
#include <vector>

#include "../../src/Redwood.hpp"
#include "KDTree.hpp"
#include "Kernel.cuh"
#include "KnnResultSet.hpp"

namespace redwood {

namespace dev {
enum class ExecutionState { kWorking, kFinished };
constexpr auto kLogLevel = 1;

// Global reference to the single KD tree
std::shared_ptr<kdt::KdTree> tree_ref;

struct ExecutorStats {
  int leaf_node_reduced = 0;
  int branch_node_reduced = 0;
};

// Pure CPU Sequential Executor
template <typename T>
class NnCpuManager {
  using QueryPointT = Point<4, T>;

 public:
  NnCpuManager(std::shared_ptr<kdt::KdTree> tree,
               const QueryPointT* my_query_points, const int my_m,
               const int tid = 0)
      : tid_(tid), tree_(tree), my_query_points_(my_query_points), my_m_(my_m) {
    tasks_todo_.resize(my_m);

    std::cout << "CpuManager " << tid_ << ":\n"
              << "\tmy_m: " << my_m_ << '\n'
              << std::endl;

    results_.resize(my_m);
    std::fill(results_.begin(), results_.end(),
              std::numeric_limits<float>::max());

    std::iota(tasks_todo_.begin(), tasks_todo_.end(), 0u);
    std::reverse(tasks_todo_.begin(), tasks_todo_.end());
  }

  void StartTraversals() {
    // Simple Sequential Traversal for CPU backend
    while (!tasks_todo_.empty()) {
      auto q_idx = tasks_todo_.back();
      tasks_todo_.pop_back();

      // redwood::StartQuery(tid_, q_idx);
      NnSearchRecursive(tree_->GetRoot(), q_idx);
    }
  }

  _NODISCARD const T GetCpuResult(const int query_idx) const {
    return results_[query_idx];
  }

  _NODISCARD ExecutorStats GetStats() const { return stats_; }

 protected:
  void NnSearchRecursive(const kdt::Node* cur, const unsigned query_idx) {
    static auto kernel_func = MyFunctor();

    if (cur->IsLeaf()) {
      const auto q = my_query_points_[query_idx];

      ++stats_.leaf_node_reduced;

      // **** Reduction at leaf node ****
      const auto leaf_size = tree_->params_.leaf_max_size;
      for (int i = 0; i < leaf_size; ++i) {
        const auto p = tree_->GetNodeContentTable()[cur->uid * leaf_size + i];
        const auto dist = kernel_func(p, q);

        results_[query_idx] = std::min(results_[query_idx], dist);
      }
      // **********************************

    } else {
      const unsigned accessor_idx = tree_->v_acc_[cur->node_type.tree.idx_mid];

      ++stats_.branch_node_reduced;

      // **** Reduction at branch node ****
      const auto dist = kernel_func(tree_->GetNodeContentTable()[accessor_idx],
                                    my_query_points_[query_idx]);
      results_[query_idx] = std::min(results_[query_idx], dist);

      // **********************************

      const auto axis = cur->node_type.tree.axis;
      const auto train = tree_->data_set_[accessor_idx].data[axis];
      const auto dir = my_query_points_[query_idx].data[axis] < train
                           ? kdt::Dir::kLeft
                           : kdt::Dir::kRight;

      NnSearchRecursive(cur->GetChild(dir), query_idx);

      const auto diff = my_query_points_[query_idx].data[axis] - train;
      if (diff * diff < results_[query_idx]) {
        NnSearchRecursive(cur->GetChild(FlipDir(dir)), query_idx);
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
  std::vector<float> results_;
};

// Basically, a pointer, an int32, a float, an int
struct CallStackField {
  kdt::Node* current;
  int axis;
  float train;
  kdt::Dir dir;
};

class NnExecutor {
  using QueryPointT = Point<4, float>;

 public:
  NnExecutor()
      : query_idx_(), q_(), state_(ExecutionState::kFinished), cur_(nullptr) {
    stack_.reserve(16);
  }

  void StartQuery(const unsigned query_idx, const QueryPointT q) {
    query_idx_ = query_idx;
    q_ = q;

    redwood::StartQuery(0, query_idx);

    cached_result_ = std::numeric_limits<float>::max();
    stack_.clear();
    cur_ = nullptr;

    Execute();
  }

  void Resume() {
    float local_result;
    redwood::GetReductionResult(0, query_idx_, &local_result);
    cached_result_ = std::min(cached_result_, local_result);
    Execute();
  }

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
          redwood::ReduceLeafNode(0, cur_->uid, query_idx_);

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
        const float dist = kernel_func(tree_ref->data_set_[accessor_idx], q_);
        cached_result_ = std::min(cached_result_, dist);

        // **********************************

        const int axis = cur_->node_type.tree.axis;
        const float train = tree_ref->data_set_[accessor_idx].data[axis];
        const kdt::Dir dir =
            q_.data[axis] < train ? kdt::Dir::kLeft : kdt::Dir::kRight;

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
        if (const auto diff = q_.data[axis] - train;
            diff * diff < cached_result_) {
          cur_ = last_cur->GetChild(FlipDir(dir));
        }
      }
    }

    // Done traversals
    state_ = ExecutionState::kFinished;
  }

  long tid_;

  // Actually essential data in a executor
  unsigned query_idx_;
  QueryPointT q_;
  std::vector<CallStackField> stack_;
  ExecutionState state_;
  kdt::Node* cur_;
  float cached_result_;
};

// -------------------------------------------------------------------------------------------------
// NN here
// -------------------------------------------------------------------------------------------------

template <typename T>
class NnExecutorManager {
  using QueryPointT = Point<4, T>;

 public:
  NnExecutorManager(std::shared_ptr<kdt::KdTree> tree,
                    const QueryPointT* my_query_points, const int my_m,
                    const int num_batches, const int tid = 0)
      : tid_(tid),
        my_query_points_(my_query_points),
        my_m_(my_m),
        num_batches_(num_batches) {
    // Save reference to
    if (!tree_ref) {
      std::cout << "[DEBUG] kdt::KdTree Reference Set!" << std::endl;
      tree_ref = tree;
    }

    assert(my_m % num_batches == 0);
    tasks_todo_.resize(my_m);

    // Need to do double buffering
    executors_.resize(num_batches);

    std::cout << "Manager " << tid_ << ":\n"
              << "\tnum queries: " << my_m_ << '\n'
              << "\tnum batches: " << num_batches_ << '\n'
              << "\tnum executors: " << executors_.size() << '\n'
              << std::endl;

    std::iota(tasks_todo_.begin(), tasks_todo_.end(), 0u);
    std::reverse(tasks_todo_.begin(), tasks_todo_.end());
  }

  void StartTraversals() {
    while (!executors_.empty()) {
      // This loop will fill the buffer.
      for (auto it = executors_.begin(); it != executors_.end();) {
        if (!it->Finished()) {
          it->Resume();
          ++it;
          continue;
        }

        if (tasks_todo_.empty()) {
          it = executors_.erase(it);
        }
        const auto q_idx = tasks_todo_.back();
        tasks_todo_.pop_back();

        it->StartQuery(q_idx, my_query_points_[q_idx]);
        ++it;
      }

      if (tasks_todo_.empty()) {
        std::cout << " Done!!!" << std::endl;
        return;  // exit(1);
      }
      // std::cout << " Tasks left:  " << tasks_todo_.size() << std::endl;

      redwood::ExecuteBatchedKernelsAsync(0, executors_.size());
    }
  }

  _NODISCARD ExecutorStats GetStats() const { return stats_; }

 protected:
 private:
  // Associated to a CUDA Stream or a SYCL queue
  int tid_;
  const Point4F* my_query_points_;
  unsigned cur_query_index_;

  // A list of local 'query_idx', with respect to global query_idx
  std::vector<unsigned> tasks_todo_;
  const int my_m_;
  const int num_batches_;

  std::vector<NnExecutor> executors_;

  ExecutorStats stats_;
};

}  // namespace dev

}  // namespace redwood