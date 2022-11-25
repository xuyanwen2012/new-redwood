// This is used to Test the Theroy
#include <algorithm>
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

// Stats
std::vector<int> num_branch_visited;
std::vector<int> num_leaf_visited;

// Consts
int stored_leaf_size;
int stored_num_threads;
int stored_num_batches;

// Shared
const Task<QueryT>* host_tasks_ref;
const DataT* host_leaf_table_ref;
const unsigned* host_leaf_sizes_ref;
std::vector<ResultT> final_results;

void InitReducer(const unsigned num_threads, const unsigned leaf_size,
                 const unsigned batch_num, const unsigned batch_size) {
  stored_num_threads = static_cast<int>(num_threads);
  stored_leaf_size = static_cast<int>(leaf_size);
  stored_num_batches = static_cast<int>(batch_num);
}

void StartQuery(const long tid, const unsigned query_idx) {}

void ReduceLeafNode(const long tid, const unsigned node_idx,
                    const unsigned query_idx) {
  constexpr auto functor = MyFunctor();

  const auto leaf_size = host_leaf_sizes_ref[query_idx];

  // std::cout << leaf_size << std::endl;
  // std::cout << host_tasks_ref[query_idx] << std::endl;

  for (int i = 0; i < leaf_size; ++i) {
    final_results[query_idx] +=
        functor(host_leaf_table_ref[node_idx * stored_leaf_size + i],
                host_tasks_ref[query_idx].query_point);
  }

  const auto start = host_leaf_table_ref + node_idx * stored_leaf_size;
  const auto end = start + leaf_size;

  ++num_leaf_visited[query_idx];
}

void ReduceBranchNode(const long tid, const void* node_element,
                      unsigned query_idx) {
  constexpr auto functor = MyFunctor();
  auto com = static_cast<const DataT*>(node_element);

  final_results[query_idx] +=
      functor(*com, host_tasks_ref[query_idx].query_point);

  ++num_branch_visited[query_idx];
}

void GetReductionResult(const long tid, const unsigned query_idx,
                        void* result) {
  const auto addr = static_cast<ResultT*>(result);
  *addr = final_results[query_idx];
}

void SetQueryPoints(long tid, const void* query_points, unsigned num_query) {
  host_tasks_ref = reinterpret_cast<const Task<QueryT>*>(query_points);

  final_results.resize(num_query);

  num_branch_visited.resize(num_query);
  num_leaf_visited.resize(num_query);
}

void SetNodeTables(const void* leaf_node_table,
                   const unsigned* leaf_node_sizes_,
                   const unsigned num_leaf_nodes) {
  host_leaf_table_ref = static_cast<const DataT*>(leaf_node_table);
  host_leaf_sizes_ref = static_cast<const unsigned*>(leaf_node_sizes_);
}

void SetBranchBatchShape(const unsigned num, const unsigned size) {}

void ExecuteBatchedKernelsAsync(long tid, const int num_batch_collected) {}

void EndReducer() {
  // for (int i = 0; i < 256; ++i) {
  //   std::cout << i << ":\tbr: " << num_branch_visited[i]
  //             << "\tle: " << num_leaf_visited[i] << std::endl;
  // }

  const auto br_max =
      *std::max_element(num_branch_visited.begin(), num_branch_visited.end());
  const auto le_max =
      *std::max_element(num_leaf_visited.begin(), num_leaf_visited.end());

  std::cout << "Br Max: " << br_max << std::endl;
  std::cout << "Le Max: " << le_max << std::endl;
}

}  // namespace redwood