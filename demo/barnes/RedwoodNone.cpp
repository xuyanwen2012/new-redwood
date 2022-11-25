// This is used to Test the Theroy
#include <algorithm>
#include <iostream>
#include <vector>

#include "../../src/Redwood.hpp"
#include "../PointCloud.hpp"
#include "Executor.hpp"

namespace redwood {

// Stats
std::vector<int> num_branch_visited;
std::vector<int> num_leaf_visited;
int leaf_reduced_counter = 0;

const unsigned* host_leaf_sizes_ref;

void InitReducer(const unsigned num_threads, const unsigned leaf_size,
                 const unsigned batch_num, const unsigned batch_size) {}

void StartQuery(long tid, const void* task_obj) {}

void ReduceLeafNode(const long tid, const unsigned node_idx,
                    const unsigned query_idx) {
  const auto leaf_size = host_leaf_sizes_ref[query_idx];
  for (int i = 0; i < leaf_size; ++i) {
    ++leaf_reduced_counter;
  }

  ++num_leaf_visited[query_idx];
}

void ReduceBranchNode(const long tid, const void* node_element,
                      unsigned query_idx) {
  ++num_branch_visited[query_idx];
}

void GetReductionResult(const long tid, const unsigned query_idx,
                        void* result) {}

void SetNodeTables(const void* leaf_node_table, const unsigned num_leaf_nodes) {
}

void SetNodeTables(const void* leaf_node_table,
                   const unsigned* leaf_node_sizes_,
                   const unsigned num_leaf_nodes) {
  host_leaf_sizes_ref = static_cast<const unsigned*>(leaf_node_sizes_);
}

void ExecuteBatchedKernelsAsync(long tid, const int num_batch_collected) {}

void SetQueryPoints(long tid, const void* query_points, unsigned num_query) {
  num_branch_visited.resize(num_query);
  num_leaf_visited.resize(num_query);
}

void EndReducer() {
  for (int i = 0; i < 32; ++i) {
    std::cout << i << ":\tbr: " << num_branch_visited[i]
              << "\tle: " << num_leaf_visited[i] << std::endl;
  }
  
  std::cout << "leaf_reduced_counter: " << leaf_reduced_counter << std::endl;

  const auto br_max =
      *std::max_element(num_branch_visited.begin(), num_branch_visited.end());
  const auto le_max =
      *std::max_element(num_leaf_visited.begin(), num_leaf_visited.end());

  std::cout << "Br Max: " << br_max << std::endl;
  std::cout << "Le Max: " << le_max << std::endl;
}

}  // namespace redwood