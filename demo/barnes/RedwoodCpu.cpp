// This is used to Test the Theroy
#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

#include "../../src/Redwood.hpp"
#include "../PointCloud.hpp"
#include "Executor.hpp"
#include "Kernel.hpp"

namespace redwood {

using DataT = Point4F;
using QueryT = Point3F;
using ResultT = Point3F;

// Consts
constexpr auto kDebugPrint = false;
int stored_leaf_size;
int stored_num_threads;
int stored_num_batches;

// Global Shared
const DataT* host_leaf_table_ref;
const unsigned* host_leaf_sizes_ref;

// Thread Local
std::vector<ResultT> final_results;
const Task<QueryT>* current_task = nullptr;

// Stats
std::vector<int> num_branch_visited;
std::vector<int> num_leaf_visited;
int leaf_reduced_counter = 0;

void InitReducer(const unsigned num_threads, const unsigned leaf_size,
                 const unsigned batch_num, const unsigned _) {
  stored_num_threads = static_cast<int>(num_threads);
  stored_leaf_size = static_cast<int>(leaf_size);
  stored_num_batches = static_cast<int>(batch_num);
}

void StartQuery(long tid, const void* task_obj) {
  const auto task = static_cast<const Task<QueryT>*>(task_obj);
  current_task = task;
}

void ReduceLeafNode(const long tid, const unsigned node_idx, const unsigned _) {
  constexpr auto functor = kernel::MyFunctor();

  const auto q = current_task->query_point;
  const auto leaf_size = host_leaf_sizes_ref[node_idx];
  for (int i = 0; i < leaf_size; ++i) {
    ++leaf_reduced_counter;
    final_results[current_task->query_idx] +=
        functor(host_leaf_table_ref[node_idx * stored_leaf_size + i], q);
  }

  if constexpr (kDebugPrint) ++num_leaf_visited[current_task->query_idx];
}

void ReduceBranchNode(const long tid, const void* node_element,
                      unsigned query_idx) {
  constexpr auto functor = kernel::MyFunctor();
  auto com = static_cast<const DataT*>(node_element);

  final_results[query_idx] += functor(*com, current_task->query_point);

  if constexpr (kDebugPrint) ++num_branch_visited[query_idx];
}

void GetReductionResult(const long tid, const unsigned query_idx,
                        void* result) {
  const auto addr = static_cast<ResultT*>(result);
  *addr = final_results[query_idx];
}

void SetQueryPoints(long tid, const void* query_points, unsigned num_query) {
  final_results.resize(num_query);

  if constexpr (kDebugPrint) {
    num_branch_visited.resize(num_query);
    num_leaf_visited.resize(num_query);
  }
}

void SetNodeTables(const void* leaf_node_table,
                   const unsigned* leaf_node_sizes_,
                   const unsigned num_leaf_nodes) {
  host_leaf_table_ref = static_cast<const DataT*>(leaf_node_table);
  host_leaf_sizes_ref = leaf_node_sizes_;
}

void ExecuteBatchedKernelsAsync(long tid, const int num_batch_collected) {}

void EndReducer() {
  if constexpr (false) {
    for (int i = 0; i < 256; ++i) {
      std::cout << i << ":\tbr: " << num_branch_visited[i]
                << "\tle: " << num_leaf_visited[i] << std::endl;
    }
  }

  if constexpr (kDebugPrint) {
    std::cout << "leaf_reduced_counter: " << leaf_reduced_counter << std::endl;

    const auto br_max =
        *std::max_element(num_branch_visited.begin(), num_branch_visited.end());
    const auto le_max =
        *std::max_element(num_leaf_visited.begin(), num_leaf_visited.end());

    std::cout << "Br Max: " << br_max << std::endl;
    std::cout << "Le Max: " << le_max << std::endl;
  }
}

}  // namespace redwood