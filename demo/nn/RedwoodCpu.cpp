#include "../../src/Redwood.hpp"

namespace redwood {

void InitReducer(const unsigned num_threads, const unsigned leaf_size,
                 const unsigned batch_num, const unsigned batch_size) {}

void StartQuery(const long tid, const unsigned query_idx) {}

void ReduceLeafNode(const long tid, const unsigned node_idx,
                    const unsigned query_idx) {}

void ReduceBranchNode(const long tid, const void* node_element,
                      unsigned query_idx) {}

void GetReductionResult(const long tid, const unsigned query_idx,
                        void* result) {}

void SetQueryPoints(const long tid, const void* query_points,
                    const unsigned num_query) {}

void SetNodeTables(const void* leaf_node_table, const unsigned num_leaf_nodes) {
}

void SetNodeTables(const void* leaf_node_table,
                   const unsigned* leaf_node_sizes_,
                   const unsigned num_leaf_nodes) {}

void SetBranchBatchShape(const unsigned num, const unsigned size) {}

void ExecuteBatchedKernelsAsync(long tid, const int num_batch_collected) {}

void EndReducer() {}

}  // namespace redwood