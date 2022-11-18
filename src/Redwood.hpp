#pragma once

namespace redwood {

// Main APIs
// New users should use the following APIs.  For GPU backends, the Preparation
// APIs are required. But FPGA backends does not require to use them.
void InitReducer(unsigned leaf_size = 32, unsigned num_threads = 1);

void StartQuery(long tid, unsigned query_idx);

void ReduceLeafNode(long tid, unsigned node_idx, unsigned query_idx);

void ReduceBranchNode(long tid, const void* node_element, unsigned query_idx);

void GetReductionResult(long tid, unsigned query_idx, void* result);

void EndReducer();

// Preparation APIs
// Only required for GPUs backends, optional for FPGA backend.
// void SetQueryPoints(const void* query_points, unsigned num_query);
void SetQueryPoints(long tid, const void* query_points, unsigned num_query);

void SetNodeTables(const void* leaf_node_table, const unsigned* leaf_node_sizes,
                   unsigned num_leaf_nodes);
void SetNodeTables(const void* leaf_node_table, unsigned num_leaf_nodes);

// Developer APIs
// Redwood developers can use the following APIs to micro controll the execution
// details. This particular function is used for GPU backend Executor Runtime.
void ExecuteBatchedKernelsAsync(long tid);

}  // namespace redwood