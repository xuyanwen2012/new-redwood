#include <CL/sycl.hpp>
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <iostream>
#include <vector>

#include "../../src/Utils.hpp"
#include "../PointCloud.hpp"
#include "../barnes/Kernel.cuh"

sycl::default_selector d_selector;

template <typename T>
using VectorAllocator = sycl::usm_allocator<T, sycl::usm::alloc::shared>;

template <typename T>
using AlignedVector = std::vector<T, VectorAllocator<T>>;

struct BatchedBuffer {
  BatchedBuffer(const sycl::queue& q, const int num, const int batch)
      : num_batch(num), batch_size(batch) {
    VectorAllocator<int> int_alloc(q);
    VectorAllocator<Point3F> p3_f_alloc(q);
    u_leaf_idx =
        std::make_shared<AlignedVector<int>>(num_batch * batch_size, int_alloc);
    u_query = std::make_shared<AlignedVector<Point3F>>(num_batch, p3_f_alloc);
    items_in_batch.resize(num_batch);
    Reset();
  }

  // Unchecked, make sure to call this no more than 'num_batches' times before
  // 'Reset()'
  void Start(const Point3F& query) {
    if (current_batch != -1) {
      items_in_batch[current_batch] = current_idx_in_batch;

      // std::fill(u_leaf_idx->begin() + current_idx_in_batch,
      //                                u_leaf_idx->begin(),
      //	Point3F{-1.0f, 0.0f, 0.0f});
    }
    ++current_batch;
    u_query->at(current_batch) = query;
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

  std::vector<int> items_in_batch;
  std::shared_ptr<AlignedVector<Point3F>> u_query;
  std::shared_ptr<AlignedVector<int>> u_leaf_idx;
};

Point3F something;

void CpuNaive(const AlignedVector<Point4F>& u_node_data, const int* u_buffer,
              const size_t size, const size_t leaf_size,
              const Point3F query_point) {
  const size_t data_size = size;
  constexpr auto functor = MyFunctor();

  Point3F sum{};

  for (int tid = 0; tid < data_size; ++tid) {
    const auto leaf_id = u_buffer[tid];
    for (int j = 0; j < leaf_size; ++j) {
      sum += functor(u_node_data[leaf_id * leaf_size + j], query_point);
    }
  }

  something += sum;
  // std::cout << "\tSUM: " << sum << std::endl;
}

void StartBatchReduction(sycl::queue& q, sycl::buffer<Point4F>& node_data_buf,
                         sycl::buffer<int>& leaf_idx_buf,
                         const size_t items_in_batch, const int leaf_size,
                         const Point3F query_point) {
  const auto data_size = items_in_batch;
  constexpr auto functor = MyFunctor();

  constexpr auto work_group_size = 256;

  const auto num_work_items = data_size;
  const auto num_work_groups = num_work_items / work_group_size;  // 1

  sycl::buffer<Point3F> accum_buf(num_work_groups);

  q.submit([&](auto& h) {
    sycl::accessor node_data_acc(node_data_buf, h, sycl::read_only);
    sycl::accessor leaf_idx_acc(leaf_idx_buf, h, sycl::read_only);
    sycl::accessor accum_acc(accum_buf, h, sycl::write_only, sycl::no_init);
    sycl::accessor<Point3F, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        scratch(work_group_size, h);
    // sycl::local_accessor<Point3F, 1> scratch(work_group_size, h);

    h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
                   [=](const sycl::nd_item<1> item) {
                     const auto global_id = item.get_global_id(0);
                     const auto local_id = item.get_local_id(0);
                     const auto group_id = item.get_group(0);

                     if (global_id < data_size) {
                       Point3F my_sum{};

                       const auto leaf_id = leaf_idx_acc[global_id];
                       for (int i = 0; i < leaf_size; ++i) {
                         my_sum +=
                             functor(node_data_acc[leaf_id * leaf_size + i],
                                     query_point);
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

                     if (local_id == 0) {
                       accum_acc[group_id] = scratch[0];
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

constexpr auto kNumStreams = 2;

int main() {
  sycl::queue q[kNumStreams];
  q[0] = sycl::queue(d_selector);
  for (int i = 1; i < kNumStreams; i++)
    q[i] = sycl::queue(q[0].get_context(), d_selector);

  auto cur_collecting = 0;

  const auto m = 1024;

  const auto num_batch = 1024;
  const auto batch_size = 1024;  // subject to change

  const auto num_leafs = 1024;
  const auto leaf_size = 256;

  VectorAllocator<Point4F> p4f_alloc(q[0]);
  AlignedVector<Point4F> u_leaf_nodes(num_leafs * leaf_size, p4f_alloc);

  // std::vector<Point4F> u_leaf_nodes(num_leafs * leaf_size);
  std::generate(u_leaf_nodes.begin(), u_leaf_nodes.end(),
                MakeRandomPoint<4, float>);

  std::vector<Point3F> q_points(m);
  std::generate(q_points.begin(), q_points.end(), MakeRandomPoint<3, float>);

  if constexpr (constexpr auto cpu = false; cpu) {
    const auto start = std::chrono::steady_clock::now();

    BatchedBuffer batched_buffer(q[0], num_batch, batch_size);

    for (int iteration = 0; iteration < 16; ++iteration) {
      // Fake Traversal
      for (int i = 0; i < num_batch; ++i) {
        batched_buffer.Start(q_points[i]);

        for (int j = 0; j < batch_size; ++j) {
          const auto id =
              static_cast<int>(my_rand(0.0f, static_cast<float>(num_leafs)));
          batched_buffer.Push(id);
        }
      }

      // Batch full, send to
      for (int batch_id = 0; batch_id < num_batch; ++batch_id) {
        CpuNaive(u_leaf_nodes, batched_buffer.GetBatchData(batch_id),
                 batch_size, leaf_size, batched_buffer.GetQuery(batch_id));
      }

      batched_buffer.Reset();
    }

    const auto end = std::chrono::steady_clock::now();
    std::cout << "CPU " << (end - start).count() << " u-secs\n";
  } else {
    WarmUp(q[0]);
    WarmUp(q[1]);

    // -----------------

    // Shared
    const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
    sycl::buffer<Point4F> node_data_buf(u_leaf_nodes.data(),
                                        u_leaf_nodes.size(), props);

    // ------------------
    // Two Batch
    BatchedBuffer batched_buffer[kNumStreams]{
        BatchedBuffer(q[0], num_batch, batch_size),
        BatchedBuffer(q[1], num_batch, batch_size),
    };

    const auto start = std::chrono::steady_clock::now();

    for (int iteration = 0; iteration < 16; ++iteration) {
      // Fake Traversal
      for (int i = 0; i < num_batch; ++i) {
        batched_buffer[cur_collecting].Start(q_points[i]);

        for (int j = 0; j < batch_size - 33; ++j) {
          const auto id =
              static_cast<int>(my_rand(0.0f, static_cast<float>(num_leafs)));
          batched_buffer[cur_collecting].Push(id);
        }
      }
      batched_buffer[cur_collecting].End();

      // Batch full, send to
      const auto num_collected =
          batched_buffer[cur_collecting].NumBatchesCollected();
      for (int batch_id = 0; batch_id < num_collected; ++batch_id) {
        sycl::buffer<int> leaf_idx_buf(
            batched_buffer[cur_collecting].GetBatchData(batch_id), batch_size,
            props);

        StartBatchReduction(
            q[cur_collecting], node_data_buf, leaf_idx_buf,
            batched_buffer[cur_collecting].ItemsInBatch(batch_id), leaf_size,
            batched_buffer[cur_collecting].GetQuery(batch_id));
      }

      // Switch
      auto next_collecting = (cur_collecting + 1) % kNumStreams;
      q[next_collecting].wait();
      cur_collecting = next_collecting;
      batched_buffer[cur_collecting].Reset();
    }

    const auto end = std::chrono::steady_clock::now();
    std::cout << "GPU " << (end - start).count() << " u-secs\n";
  }

  // ------------------

  std::cout << something << std::endl;
  return 0;
}
