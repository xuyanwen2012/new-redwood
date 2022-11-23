// This is used to Test the Theroy
#include <algorithm>
#include <iostream>
#include <vector>

#include "../../src/Redwood.hpp"

namespace redwood {

// Stats
std::vector<int> num_branch_visited;
std::vector<int> num_leaf_visited;

void InitReducer(const unsigned num_threads, const unsigned leaf_size,
                 const unsigned batch_num, const unsigned batch_size) {}

void StartQuery(const long tid, const unsigned query_idx) {}

void ReduceLeafNode(const long tid, const unsigned node_idx,
                    const unsigned query_idx) {
  ++num_leaf_visited[query_idx];
}

void ReduceBranchNode(const long tid, const void* node_element,
                      unsigned query_idx) {
  ++num_branch_visited[query_idx];
}

void GetReductionResult(const long tid, const unsigned query_idx,
                        void* result) {}

void SetQueryPoints(const long tid, const void* query_points,
                    const unsigned num_query) {
  num_branch_visited.resize(num_query);
  num_leaf_visited.resize(num_query);
}

void SetNodeTables(const void* leaf_node_table, const unsigned num_leaf_nodes) {
}

void SetNodeTables(const void* leaf_node_table,
                   const unsigned* leaf_node_sizes_,
                   const unsigned num_leaf_nodes) {}

void SetBranchBatchShape(const unsigned num, const unsigned size) {}

void ExecuteBatchedKernelsAsync(long tid, const int num_batch_collected) {}

void EndReducer() {
  for (int i = 0; i < 1024; ++i) {
    std::cout << i << ":\tbr: " << num_branch_visited[i]
              << "\tle: " << num_leaf_visited[i] << std::endl;
  }

  const auto br_max =
      *std::max_element(num_branch_visited.begin(), num_branch_visited.end());
  const auto le_max =
      *std::max_element(num_leaf_visited.begin(), num_leaf_visited.end());

  std::cout << "Br Max: " << br_max << std::endl;
  std::cout << "Le Max: " << le_max << std::endl;
}

}  // namespace redwood