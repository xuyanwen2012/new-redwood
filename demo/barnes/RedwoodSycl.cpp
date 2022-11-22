#include <sys/types.h>

#include <CL/sycl.hpp>
#include <memory>
#include <vector>

#include "../PointCloud.hpp"
#include "Kernel.cuh"

namespace redwood {

struct BatchedBuffer;

template <typename T>
using VectorAllocator = sycl::usm_allocator<T, sycl::usm::alloc::shared>;

template <typename T>
using AlignedVector = std::vector<T, VectorAllocator<T>>;

// ------------------- Constants -------------------
constexpr auto kBlockThreads = 256;
constexpr auto kNumStreams = 2;
int stored_leaf_size;
int stored_num_threads;
int stored_num_batches;  // num_batches == num_blocks == num_executors
int stored_batch_size;   // better to be multiple of 'num_threads'

sycl::device device;
sycl::context ctx;
sycl::property_list props;

// ------------------- Variables -------------------

sycl::queue q[kNumStreams];
int cur_collecting;

// Two Batch
std::vector<BatchedBuffer> batched_buffer;

// ------------------- Shared Data -------------------

std::vector<Point3F> q_points;
std::unique_ptr<sycl::buffer<Point4F>> b_node_data_buf;

// Stored the results of all queries
std::vector<Point3F> le_results;
std::unique_ptr<sycl::buffer<Point3F>> b_result_buf;

// -------------- Buffer related -------------

struct BatchedBuffer {
  BatchedBuffer(const sycl::queue& q, const int num, const int batch)
      : num_batch(num), batch_size(batch) {
    VectorAllocator<int> int_alloc(q);
    VectorAllocator<Point3F> p3_f_alloc(q);
    u_leaf_idx =
        std::make_shared<AlignedVector<int>>(num_batch * batch_size, int_alloc);
    u_query = std::make_shared<AlignedVector<Point3F>>(num_batch, p3_f_alloc);
    items_in_batch.resize(num_batch);
    query_indx.resize(num_batch);
    Reset();
  }

  // Unchecked, make sure to call this no more than 'num_batches' times before
  // 'Reset()'
  void Start(const int q_idx) {
    if (current_batch != -1) {
      items_in_batch[current_batch] = current_idx_in_batch;
    }
    ++current_batch;
    u_query->at(current_batch) = q_points[q_idx];
    query_indx[current_batch] = q_idx;
    current_idx_in_batch = 0;
  }

  // Unchecked, make sure to call this no more than 'leaf_size' times before
  // next 'Start()'
  void Push(const int leaf_id) {
    u_leaf_idx->at(current_batch * batch_size + current_idx_in_batch) = leaf_id;
    ++current_idx_in_batch;
  }

  void End() {
    items_in_batch[current_batch] = current_idx_in_batch;
    ++current_batch;
  }

  void Reset() {
    current_batch = -1;
    current_idx_in_batch = 0;

    // TODO: optimize
    std::fill(u_query->begin(), u_query->end(), Point3F{-1.0f, 0.0f, 0.0f});
    std::fill(u_leaf_idx->begin(), u_leaf_idx->end(), 0);
    std::fill(items_in_batch.begin(), items_in_batch.end(), 0);
    std::fill(query_indx.begin(), query_indx.end(), 0);
  }

  _NODISCARD int* GetBatchData(const int batch_id) const {
    return &u_leaf_idx->at(batch_id * batch_size);
  }

  _NODISCARD Point3F GetQuery(const int batch_id) const {
    return u_query->at(batch_id);
  }

  _NODISCARD int NumBatchesCollected() const { return current_batch; }
  _NODISCARD int ItemsInBatch(const int batch_id) const {
    return items_in_batch[batch_id];
  }

  int current_idx_in_batch;
  int current_batch;

  int num_batch;
  int batch_size;

  //
  std::vector<int> query_indx;
  std::vector<int> items_in_batch;

  // Part of the buffer data
  std::shared_ptr<AlignedVector<Point3F>> u_query;
  std::shared_ptr<AlignedVector<int>> u_leaf_idx;
};

// -------------- SYCL related -------------

void StartBatchReduction(sycl::queue& q, sycl::buffer<Point4F>& node_data_buf,
                         sycl::buffer<int>& leaf_idx_buf, const int query_idx,
                         const size_t items_in_batch, const int leaf_size,
                         const Point3F query_point) {
  const auto data_size = items_in_batch;

  constexpr auto functor = MyFunctor();

  constexpr auto work_group_size = kBlockThreads;  // 256

  const auto num_work_items = data_size;

  // const auto num_work_groups = num_work_items / work_group_size;
  // sycl::buffer<Point3F> accum_buf(num_work_groups);

  q.submit([&](auto& h) {
    sycl::accessor node_data_acc(node_data_buf, h, sycl::read_only);
    sycl::accessor leaf_idx_acc(leaf_idx_buf, h, sycl::read_only);
    sycl::accessor accum_acc(*b_result_buf, h, sycl::read_write);
    // sycl::accessor accum_acc(accum_buf, h, sycl::write_only, sycl::no_init);
    sycl::local_accessor<Point3F, 1> scratch(work_group_size, h);

    h.parallel_for(
        sycl::nd_range<1>(num_work_items, work_group_size),
        [=](const sycl::nd_item<1> item) {
          const auto global_id = item.get_global_id(0);
          const auto local_id = item.get_local_id(0);
          const auto group_id = item.get_group(0);

          if (global_id < data_size) {
            Point3F my_sum{};

            const auto leaf_id = leaf_idx_acc[global_id];
            for (int i = 0; i < leaf_size; ++i) {
              my_sum +=
                  functor(node_data_acc[leaf_id * leaf_size + i], query_point);
            }

            scratch[local_id] = my_sum;
          } else {
            scratch[local_id] = Point3F();
          }

          // Do a tree reduction on items in work-group
          for (int i = work_group_size / 2; i > 0; i >>= 1) {
            item.barrier(sycl::access::fence_space::local_space);
            if (local_id < i) {
              scratch[local_id] += scratch[local_id + i];
            }
          }

          if (global_id == 0) {
            // Test
            // auto v_x =
            //     sycl::atomic_ref<float,
            //     sycl::memory_order::relaxed,
            //                      sycl::memory_scope::device,
            //                      sycl::access::address_space::global_space>(
            //         accum_acc[batch_id].data[0]);
            // v_x.fetch_add(scratch[0].data[0]);

            accum_acc[query_idx] = Point3F{scratch[0].data[0], 666.f, 666.f};
          }
        });
  });
}

void WarmUp(sycl::queue& q) {
  int sum;
  sycl::buffer<int> sum_buf(&sum, 1);
  q.submit([&](auto& h) {
    sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);
    h.parallel_for(1, [=](auto) { sum_acc[0] = 0; });
  });
  q.wait();
}

// -------------- REDwood APIs -------------
void InitReducer(const unsigned num_threads, const unsigned leaf_size,
                 const unsigned batch_num, const unsigned batch_size) {
  // Save Parameters
  stored_num_threads = num_threads;
  stored_leaf_size = leaf_size;
  stored_num_batches = batch_num;
  stored_batch_size = batch_size;

  // Set up Sycl
  // Intel(R) UHD Graphics [0x9b41] on Parakeet
  device = sycl::device::get_devices(sycl::info::device_type::all)[1];
  std::cout << "SYCL Device: " << device.get_info<sycl::info::device::name>()
            << std::endl;
  props = {sycl::property::buffer::use_host_ptr()};

  q[0] = sycl::queue(device);
  for (int i = 1; i < kNumStreams; i++)
    q[i] = sycl::queue(q[0].get_context(), device);

  // Setup
  for (int i = 0; i < kNumStreams; i++)
    batched_buffer.emplace_back(q[i], batch_num, batch_size);

  for (int i = 0; i < kNumStreams; i++) WarmUp(q[0]);

  cur_collecting = 0;
}

void StartQuery(const long tid, const unsigned query_idx) {
  batched_buffer[cur_collecting].Start(query_idx);
}

void ReduceLeafNode(const long tid, const unsigned node_idx,
                    const unsigned query_idx) {
  batched_buffer[cur_collecting].Push(node_idx);
}

void ReduceBranchNode(const long tid, const void* node_element,
                      unsigned query_idx) {
  auto kernel_func = MyFunctor();
  const auto p = static_cast<const Point4F*>(node_element);

  // results[tid][query_idx] += kernel_func(*p,
  // query_data_base[tid][query_idx]);
}

void GetReductionResult(const long tid, const unsigned query_idx,
                        void* result) {
  // TODO: Add offset
  auto addr = static_cast<Point3F*>(result);
}

void SetQueryPoints(const long tid, const void* query_points,
                    const unsigned num_query) {
  auto ptr = static_cast<const Point3F*>(query_points);
  q_points.assign(ptr, ptr + num_query);

  // Allocate Result Buffer (no 'use_host_ptr')
  le_results.resize(num_query);
  b_result_buf = std::make_unique<sycl::buffer<Point3F>>(
      le_results.data(), sycl::range{num_query});
}

void SetNodeTables(const void* leaf_node_table, const unsigned num_leaf_nodes) {
  auto lnd = static_cast<const Point4F*>(leaf_node_table);
  b_node_data_buf = std::make_unique<sycl::buffer<Point4F>>(
      lnd, sycl::range{num_leaf_nodes * stored_leaf_size}, props);
}

void SetNodeTables(const void* leaf_node_table,
                   const unsigned* leaf_node_sizes_,
                   const unsigned num_leaf_nodes) {
  SetNodeTables(leaf_node_table, num_leaf_nodes);
}

void ExecuteBatchedKernelsAsync(long tid) {
  // Terminate the Current Batch
  batched_buffer[cur_collecting].End();

  const auto num_collected =
      batched_buffer[cur_collecting].NumBatchesCollected();

  // Used to store the intermediate results
  // constexpr auto accum_buffer_size = stored_batch_size / kBlockThreads;
  // std::vector<Point3F> tmp_results(num_collected);
  // sycl::buffer<Point3F> accum_buf(num_collected);

  // Process each batch as independent Kernel
  for (int batch_id = 0; batch_id < num_collected; ++batch_id) {
    const auto items_in_batch =
        batched_buffer[cur_collecting].ItemsInBatch(batch_id);

    if (items_in_batch == 0) continue;

    sycl::buffer<int> leaf_idx_buf(
        batched_buffer[cur_collecting].GetBatchData(batch_id),
        stored_batch_size, props);

    // q[cur_collecting].submit([&](auto& h) {
    //   sycl::accessor node_data_acc(*b_node_data_buf, h, sycl::read_only);
    //   sycl::accessor leaf_idx_acc(leaf_idx_buf, h, sycl::read_only);
    //   sycl::accessor accum_acc(accum_buf, h, sycl::read_write);
    //   // sycl::accessor accum_acc(accum_buf, h, sycl::write_only,
    //   // sycl::no_init);
    //   sycl::local_accessor<Point3F, 1> scratch(256, h);

    //   int num_work_items = MyRoundUp(items_in_batch, 256);

    //   const auto local_batch_id = batch_id;
    //   h.parallel_for(
    //       sycl::nd_range<1>(num_work_items, 256),
    //       [=](const sycl::nd_item<1> item) {
    //         const auto global_id = item.get_global_id(0);

    //         if (global_id == 0) {
    //           accum_acc[local_batch_id] = Point3F{666.f, 666.f, 666.f};
    //         }
    //       });
    // });
    // q[cur_collecting].wait();

    StartBatchReduction(q[cur_collecting], *b_node_data_buf, leaf_idx_buf,
                        stored_batch_size, stored_leaf_size,
                        batched_buffer[cur_collecting].query_indx[batch_id],
                        batched_buffer[cur_collecting].GetQuery(batch_id));
  }

  // q[0].wait();

  sycl::host_accessor h_result_acc(*b_node_data_buf);
  for (int i = 0; i < 14; ++i) {
    std::cout << batched_buffer[cur_collecting].query_indx[i] << ": "
              << h_result_acc[i] << "\t" << le_results[i] << std::endl;
  }

  exit(1);
  // Switch
  auto next_collecting = (cur_collecting + 1) % kNumStreams;
  q[next_collecting].wait();
  cur_collecting = next_collecting;
  batched_buffer[cur_collecting].Reset();
}

void EndReducer() { q[cur_collecting].wait(); }

}  // namespace redwood