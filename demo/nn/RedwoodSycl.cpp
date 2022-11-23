#include <CL/sycl.hpp>
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <vector>

#include "../../src/sycl/Utils.hpp"
#include "../PointCloud.hpp"
#include "Kernel.cuh"
#include "KnnResultSet.hpp"

namespace redwood {

// ------------------- Constants -------------------
constexpr auto kBlockThreads = 256;
constexpr auto kNumStreams = 1;

int stored_leaf_size;
int stored_num_threads;
int stored_num_batches;  // num_batches == num_work_groups == num_executors

sycl::device device;
sycl::context ctx;
// sycl::property_list props;

// ------------------- Variables -------------------

sycl::queue q[kNumStreams];
int cur_collecting;

// ------------------- Shared Data -------------------

const Point4F* q_points_ref;
std::unique_ptr<sycl::buffer<Point4F>> queries_table_buf;

const Point4F* lnd_ref;
std::unique_ptr<sycl::buffer<Point4F>> leaf_node_table_buf;

// -------------- Buffer related -------------

struct NnBuffer {
  NnBuffer(const sycl::queue& q, const int buffer_size) {
    query_idx = std::make_unique<UsmVector<int>>(buffer_size, IntAlloc(q));
    leaf_idx = std::make_unique<UsmVector<int>>(buffer_size, IntAlloc(q));

    next_avalible_slot = 0;
  }

  void AllocateResult(const sycl::queue& q, const int m) {
    results = std::make_unique<UsmVector<float>>(m, FloatAlloc(q));
    std::fill(results->begin(), results->end(),
              std::numeric_limits<float>::max());
  }

  _NODISCARD size_t Size() const { return query_idx->size(); }

  void Reset() { next_avalible_slot = 0; }

  void LoadLeafNode(const unsigned q_idx, const unsigned node_idx) {
    // u_buffer[next_avalible_slot] = make_uint2(q_idx, node_idx);
    query_idx->operator[](next_avalible_slot) = q_idx;
    leaf_idx->operator[](next_avalible_slot) = node_idx;

    ++next_avalible_slot;
  }

  std::unique_ptr<UsmVector<int>> query_idx;  // buffer_size
  std::unique_ptr<UsmVector<int>> leaf_idx;   // buffer_size
  std::unique_ptr<UsmVector<float>> results;  // m for now

  // Misc
  int next_avalible_slot;
};

std::vector<NnBuffer> nn_buffers;

void StartProcessBufferNaive(sycl::queue& q, NnBuffer& buffer) {
  constexpr auto kernel_func = MyFunctor();

  const auto query_idx_ptr = buffer.query_idx->data();
  const auto leaf_idx_ptr = buffer.leaf_idx->data();
  const auto results_ptr = buffer.results->data();
  const auto leaf_size = stored_leaf_size;
  q.submit([&](sycl::handler& h) {
    const sycl::accessor leaf_table_acc(*leaf_node_table_buf, h,
                                        sycl::read_only);
    const sycl::accessor query_table_acc(*queries_table_buf, h,
                                         sycl::read_only);
    h.parallel_for(sycl::range(buffer.Size()), [=](const sycl::id<1> idx) {
      const auto query_id = query_idx_ptr[idx];
      const auto leaf_id = leaf_idx_ptr[idx];

      const auto q_point = query_table_acc[query_id];
      auto my_min = std::numeric_limits<float>::max();
      for (int i = 0; i < leaf_size; ++i) {
        const auto dist =
            kernel_func(leaf_table_acc[leaf_id * leaf_size + i], q_point);
        my_min = sycl::min(my_min, dist);
      }

      results_ptr[query_id] = sycl::min(results_ptr[query_id], my_min);
    });
  });

  // q.wait();
}

// -------------- REDwood APIs -------------
void InitReducer(const unsigned num_threads, const unsigned leaf_size,
                 const unsigned batch_num, const unsigned batch_size) {
  // Save Parameters
  stored_num_threads = num_threads;
  stored_leaf_size = leaf_size;
  stored_num_batches = batch_num;

  // Set up Sycl

  ShowDevice(q[0]);

  // Intel(R) UHD Graphics [0x9b41] on Parakeet
  device = sycl::device::get_devices(sycl::info::device_type::all)[1];

  q[0] = sycl::queue(device);
  for (int i = 1; i < kNumStreams; i++)
    q[i] = sycl::queue(q[0].get_context(), device);

  // Setup

  // A Two buffers (Double buffering)
  for (int i = 0; i < kNumStreams; i++)
    nn_buffers.emplace_back(q[i], batch_num);

  for (int i = 0; i < kNumStreams; i++) WarmUp(q[i]);

  cur_collecting = 0;
}

void StartQuery(const long tid, const unsigned query_idx) {}

void ReduceLeafNode(const long tid, const unsigned node_idx,
                    const unsigned query_idx) {
  nn_buffers[cur_collecting].LoadLeafNode(query_idx, node_idx);
}

void ReduceBranchNode(const long tid, const void* node_element,
                      unsigned query_idx) {
  constexpr auto kernel_func = MyFunctor();
  const auto p = static_cast<const Point4F*>(node_element);
  auto dist = kernel_func(*p, q_points_ref[query_idx]);
}

void GetReductionResult(const long tid, const unsigned query_idx,
                        void* result) {
  auto addr = static_cast<float*>(result);
  *addr = nn_buffers[cur_collecting].results->operator[](query_idx);
}

void SetQueryPoints(const long tid, const void* query_points,
                    const unsigned num_query) {
  q_points_ref = static_cast<const Point4F*>(query_points);
  queries_table_buf = std::make_unique<sycl::buffer<Point4F>>(
      q_points_ref, sycl::range{num_query});

  // Set results
  for (int i = 0; i < kNumStreams; ++i) {
    nn_buffers[i].AllocateResult(q[i], num_query);
  }
}

void SetNodeTables(const void* leaf_node_table, const unsigned num_leaf_nodes) {
  auto lnd = static_cast<const Point4F*>(leaf_node_table);
  lnd_ref = lnd;
  leaf_node_table_buf = std::make_unique<sycl::buffer<Point4F>>(
      lnd, sycl::range{num_leaf_nodes * stored_leaf_size});
}

void SetNodeTables(const void* leaf_node_table,
                   const unsigned* leaf_node_sizes_,
                   const unsigned num_leaf_nodes) {
  SetNodeTables(leaf_node_table, num_leaf_nodes);
}

void ExecuteBatchedKernelsAsync(long tid, const int num_batch_collected) {
  StartProcessBufferNaive(q[cur_collecting], nn_buffers[cur_collecting]);
  const auto next_collecting = (cur_collecting + 1) % kNumStreams;
  q[next_collecting].wait();
  cur_collecting = next_collecting;
  nn_buffers[cur_collecting].Reset();
}

void EndReducer() {
  const auto next_collecting = (cur_collecting + 1) % kNumStreams;
  q[next_collecting].wait();
}

}  // namespace redwood