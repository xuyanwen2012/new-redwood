#pragma once

namespace redwood {
// Main APIs
// New users should use the following APIs.  For GPU backends, the Preparation
// APIs are required. But FPGA backends does not require to use them.
void InitReducer(unsigned num_threads = 1, unsigned leaf_size = 32,
                 unsigned batch_num = 1024, unsigned batch_size = 1024);

void StartQuery(long tid, unsigned query_idx);
void StartQuery(long tid, const void* task_obj);

void ReduceLeafNode(long tid, unsigned node_idx, unsigned query_idx);

void ReduceBranchNode(long tid, const void* node_element, unsigned query_idx);
// void ReduceBranchNode(long tid, const void* node_element, const void*
// task_obj);

void GetReductionResult(long tid, unsigned query_idx, void* result);

void EndReducer();

// Preparation APIs
// Only required for GPUs backends, optional for FPGA backend.
// void SetQueryPoints(const void* query_points, unsigned num_query);
// void SetQueryPoints(long tid, const void* query_points, unsigned num_query);
void SetQueryPoints(long tid, const void* query_points, unsigned num_query);

void SetNodeTables(const void* leaf_node_table, const unsigned* leaf_node_sizes,
                   unsigned num_leaf_nodes);
void SetNodeTables(const void* leaf_node_table, unsigned num_leaf_nodes);

// Developer APIs
// Redwood developers can use the following APIs to micro controll the execution
// details. This particular function is used for GPU backend Executor Runtime.
void ExecuteBatchedKernelsAsync(long tid, int num_batch_collected);

void ReduceLeafNodeWithTask(long tid, unsigned node_idx, const void* task);

}  // namespace redwood
