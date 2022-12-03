#include <fcntl.h>
#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <limits>

#include "../../src/Redwood.hpp"
#include "../PointCloud.hpp"

using DataT = Point2D;
using QueryT = Point2D;
using ResultT = double;  // Special for KNN

constexpr auto kArg = 0;
constexpr auto kPos0X = 1;
constexpr auto kPos0Y = 2;
constexpr auto kFincnt = 3;
constexpr auto kResult = 4;
constexpr auto kNEngine = 1;
constexpr auto kDuetLeafSize = 64;

volatile uint64_t* duet_baseaddr = nullptr;

int stored_leaf_size;

namespace redwood {
void InitReducer(const unsigned num_threads, const unsigned leaf_size,
                 const unsigned batch_num, const unsigned batch_size) {
  // stored_num_threads = static_cast<int>(num_threads);
  stored_leaf_size = static_cast<int>(leaf_size);
  // stored_num_batches = static_cast<int>(batch_num);

  if constexpr (kRedwoodBackend != redwood::Backends::kDuet) {
    // 8k?
    int fd = open("/dev/duet", O_RDWR);
    duet_baseaddr = static_cast<volatile uint64_t*>(mmap(
        nullptr, kNEngine << 13, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0));
  } else {
    duet_baseaddr = static_cast<volatile uint64_t*>(malloc(kNEngine << 13));
  }
}

void StartQuery(const long tid, const void* query_element) {
  auto ptr = reinterpret_cast<const Point2D*>(query_element);

  const long caller_id = tid;
  volatile uint64_t* sri = duet_baseaddr + (caller_id << 4) + 16;

  if constexpr (false) {
    std::cout << tid << ": started duet. " << ptr->data[0] << std::endl;
  }

  sri[kPos0X] = *reinterpret_cast<const uint64_t*>(&ptr->data[0]);
  sri[kPos0Y] = *reinterpret_cast<const uint64_t*>(&ptr->data[1]);
}

inline void ReduceLeafNode(const long tid, const void* node_base_addr) {
  const long caller_id = tid;
  volatile uint64_t* sri = duet_baseaddr + (caller_id << 4) + 16;

  if constexpr (false) {
    auto ptr = reinterpret_cast<const Point2D*>(node_base_addr);
    std::cout << tid << ": pushed duet. " << ptr->data[0]
              << "\taddress: " << node_base_addr << std::endl;
  }

  sri[kArg] = reinterpret_cast<uint64_t>(node_base_addr);
}

void ReduceLeafNode(const long tid, const unsigned node_idx,
                    const unsigned query_idx) {
  if constexpr (false) {
    std::cout << tid << ": ReduceLeafNode, node id:  " << node_idx << std::endl;
  }

  // Must set 'leaf_nodes_data' before calling this function

  for (int i = 0; i < stored_leaf_size; i += kDuetLeafSize) {
    // Each call takes the next 'kDuetLeafSize' elements from that base pointer.
    ReduceLeafNode(tid, &leaf_nodes_data[node_idx * stored_leaf_size + i]);
  }
}

void GetReductionResult(const long tid, const unsigned query_idx,
                        void* result) {
  const long caller_id = tid;
  volatile uint64_t* sri = duet_baseaddr + (caller_id << 4) + 16;

  if constexpr (false) {
    std::cout << tid << ": poping. " << std::endl;
  }

  while (sri[kFincnt] < reduction_iteration) NO_OP;

  auto addr = static_cast<double*>(result);
  auto tmp = *reinterpret_cast<const volatile double*>(&sri[kResult]);

  *addr = tmp;
}

void SetQueryPoints(const long tid, const void* query_points,
                    const unsigned num_query) {
  const auto total_count = num_query;
}

void SetNodeTables(const void* leaf_node_table, const unsigned num_leaf_nodes) {
  host_leaf_node_table_ref = static_cast<const DataT*>(leaf_node_table);
  leaf_node_table_buf = std::make_unique<sycl::buffer<DataT>>(
      host_leaf_node_table_ref, sycl::range{num_leaf_nodes * stored_leaf_size});
}

void ExecuteBatchedKernelsAsync(long tid, const int num_batch_collected) {

}

void ReduceLeafNodeWithTask(long tid, const unsigned node_idx,
                            const void* task) {
  // const auto t = static_cast<const Task*>(task);
  // nn_buffers[cur_collecting].PushNewTask(*t, node_idx);
}

void EndReducer() {}
}  // namespace redwood
