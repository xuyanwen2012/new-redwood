#include <CL/sycl.hpp>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../../src/Redwood.hpp"
#include "../PointCloud.hpp"
#include "Kernel.cuh"

namespace redwood {

// ------------------- Constants -------------------
constexpr auto kNumThreads = 256;
constexpr auto kNumStreams = 2;
int stored_leaf_size;
int stored_num_batches;
int stored_num_threads;

// -------------------  Global Variables -------------------
sycl::device device;
sycl::context ctx;
sycl::property_list props;

unsigned cur_collecting;
sycl::queue streams[kNumStreams];
// BarnesBranchBatch br_batches[kNumStreams];
static inline sycl::queue& CurrentStream() { return streams[cur_collecting]; }
// static inline BarnesBranchBatch& CurrentBranchBatch() {
// return br_batches[cur_collecting];
// }

// -------------------  SYCL Related -------------------

class MyReduction;

void InitSycl() {
  // Intel(R) UHD Graphics [0x9b41] on Parakeet
  device = sycl::device::get_devices(sycl::info::device_type::all)[1];
  std::cout << "SYCL Device: " << device.get_info<sycl::info::device::name>()
            << std::endl;

  props = {sycl::property::buffer::use_host_ptr()};

  streams[0] = sycl::queue(device);
  for (int i = 1; i < kNumStreams; i++) {
    streams[i] = sycl::queue(streams[0].get_context(), device);
  }

  cur_collecting = 0;
}

// -------------------  Application Related -------------------

// Shared Data
// const Point4F* leaf_nodes_data;
const unsigned* leaf_node_sizes;
sycl::buffer<Point4F>* b_node_content_table;

// Thread local
std::vector<Point3F*> u_results;
std::vector<sycl::buffer<Point3F>*> b_query_data;

namespace internal {

void AllocateAndCopyLeafNodesData(const void* leaf_node_table,
                                  const unsigned num_leaf_nodes) {
  auto lnd = static_cast<const Point4F*>(leaf_node_table);
  b_node_content_table = new sycl::buffer(
      lnd, sycl::range{num_leaf_nodes * stored_leaf_size}, props);
}

void AllocateAndCopyQueryData(const long tid, const void* query_data,
                              const unsigned m) {
  auto qd = static_cast<const Point3F*>(query_data);
  b_query_data[tid] = new sycl::buffer(qd, sycl::range{m}, props);
}

void AllocateResults(const long tid, const unsigned m) {
  const auto bytes = sizeof(Point3F) * m;
  u_results[tid] =
      static_cast<Point3F*>(sycl::malloc_shared(bytes, device, ctx, props));

  std::fill_n(u_results[tid], m, Point3F());
}

}  // namespace internal

// std::vector<const Point3F*> query_data_base;
// std::vector<Point3F*> u_results;

// -------------------  REDwood APIs -------------------

void InitReducer(const unsigned leaf_size, const unsigned num_threads) {
  stored_leaf_size = leaf_size;
  stored_num_threads = num_threads;

  InitSycl();

  for (int i = 0; i < kNumStreams; ++i) {
    // batches[i].AllocateBuffer(kNumBatches);
  }

  b_query_data.resize(num_threads);
  u_results.resize(num_threads);
}

void StartQuery(const long tid, const unsigned query_idx) {}

void ReduceLeafNode(const long tid, const unsigned node_idx,
                    const unsigned query_idx) {}

void ReduceBranchNode(const long tid, const void* node_element,
                      unsigned query_idx) {
  // auto kernel_func = MyFunctor();
  // const auto p = static_cast<const Point4F*>(node_element);
  // results[tid][query_idx] += kernel_func(*p,
  // query_data_base[tid][query_idx]);

  // if constexpr (true) {
  //   std::cout << "\tresults[tid][query_idx]: " << results[tid][query_idx]
  //             << " *p:\t" << *p << "\tquery_data_base[tid][query_idx]"
  //             << query_data_base[tid][query_idx] << std::endl;
  // }
}

void GetReductionResult(const long tid, const unsigned query_idx,
                        void* result) {
  auto addr = static_cast<Point3F*>(result);
  *addr = u_results[tid][query_idx];
}

void SetQueryPoints(const long tid, const void* query_points,
                    const unsigned num_query) {
  internal::AllocateAndCopyQueryData(tid, query_points, num_query);
  internal::AllocateResults(tid, num_query);
}

void SetNodeTables(const void* leaf_node_table, const unsigned num_leaf_nodes) {
  internal::AllocateAndCopyLeafNodesData(leaf_node_table, num_leaf_nodes);
}

void SetNodeTables(const void* leaf_node_table,
                   const unsigned* leaf_node_sizes_,
                   const unsigned num_leaf_nodes) {
  SetNodeTables(leaf_node_table, num_leaf_nodes);
  leaf_node_sizes = leaf_node_sizes_;
}

void ExecuteBatchedKernelsAsync(long tid) {
  // ExecuteKernel(CurrentBatch());
  // auto next_collecting = (cur_collecting + 1) % kNumStreams;
  // streams[next_collecting].wait();
  // cur_collecting = next_collecting;
  // CurrentBatch().Reset();
}

void EndReducer() { CurrentStream().wait(); }

}  // namespace redwood