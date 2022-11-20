#include <cstdlib>
#include <cstring>
#include <vector>

#include "../../src/Redwood.hpp"
#include "../PointCloud.hpp"
#include "Kernel.cuh"

namespace redwood {
constexpr auto kDebugPrint = false;

// Constants
unsigned stored_leaf_size;
unsigned stored_num_treads;

// Shared Data
const Point4F* leaf_nodes_data;
const unsigned* leaf_node_sizes;

// Thread Local Data
std::vector<const Point3F*> query_data_base;
std::vector<std::vector<Point3F>> results;

void InitReducer(const unsigned num_threads, const unsigned leaf_size,
                 const unsigned batch_num, const unsigned batch_size) {
  stored_num_treads = num_threads;
  stored_leaf_size = leaf_size;

  query_data_base.resize(num_threads);
  results.resize(num_threads);
}

void StartQuery(const long tid, const unsigned query_idx) {}

void ReduceLeafNode(const long tid, const unsigned node_idx,
                    const unsigned query_idx) {
  // Must set 'leaf_nodes_data' before calling this function
  if constexpr (kDebugPrint) {
    std::cout << tid << ": ReduceLeafNode, node id:  " << node_idx << std::endl;
  }

  Point3F sum{};
  auto kernel_func = MyFunctor();

  const auto num = leaf_node_sizes[node_idx];
  for (auto i = 0u; i < num; ++i) {
    sum += kernel_func(leaf_nodes_data[node_idx * stored_leaf_size + i],
                       query_data_base[tid][query_idx]);
  }

  results[tid][query_idx] += sum;
}

void ReduceBranchNode(const long tid, const void* node_element,
                      unsigned query_idx) {
  if constexpr (kDebugPrint) {
    std::cout << tid << ": ReduceBranchNode, node id:  " << node_element
              << std::endl;
  }

  auto kernel_func = MyFunctor();
  const auto p = static_cast<const Point4F*>(node_element);
  results[tid][query_idx] += kernel_func(*p, query_data_base[tid][query_idx]);

  // if constexpr (true) {
  //   std::cout << "\tresults[tid][query_idx]: " << results[tid][query_idx]
  //             << " *p:\t" << *p << "\tquery_data_base[tid][query_idx]"
  //             << query_data_base[tid][query_idx] << std::endl;
  // }
}

void GetReductionResult(const long tid, const unsigned query_idx,
                        void* result) {
  auto addr = static_cast<Point3F*>(result);
  *addr = results[tid][query_idx];
}

void SetQueryPoints(const long tid, const void* query_points,
                    const unsigned num_query) {
  query_data_base[tid] = static_cast<const Point3F*>(query_points);
  results[tid].resize(num_query);
}

void SetNodeTables(const void* leaf_node_table, const unsigned num_leaf_nodes) {
  leaf_nodes_data = static_cast<const Point4F*>(leaf_node_table);
}

void SetNodeTables(const void* leaf_node_table,
                   const unsigned* leaf_node_sizes_,
                   const unsigned num_leaf_nodes) {
  SetNodeTables(leaf_node_table, num_leaf_nodes);
  leaf_node_sizes = leaf_node_sizes_;
}

void SetBranchBatchShape(const unsigned num, const unsigned size) {}

void ExecuteBatchedKernelsAsync(long tid) {}

void EndReducer() {}

}  // namespace redwood