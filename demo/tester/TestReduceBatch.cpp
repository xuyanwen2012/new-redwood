//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <sycl/sycl.hpp>
#include <vector>

#include "../../src/Utils.hpp"
#include "../PointCloud.hpp"
#include "../barnes/Kernel.cuh"

sycl::default_selector d_selector;

template <typename T>
using VectorAllocator = sycl::usm_allocator<T, sycl::usm::alloc::shared>;

template <typename T>
using AlignedVector = std::vector<T, VectorAllocator<T>>;

void CpuNaive(const AlignedVector<Point4F> &u_data,
              const AlignedVector<Point3F> &u_query, const int num_batches,
              const int batch_size) {
  const size_t data_size = num_batches * batch_size;
  auto functor = MyFunctor();

  std::vector<Point3F> sums(num_batches);

  auto start = std::chrono::steady_clock::now();

  for (int tid = 0; tid < data_size; ++tid) {
    const auto batch_id = tid / batch_size;
    sums[batch_id] += functor(u_data[tid], u_query[batch_id]);
  }

  auto end = std::chrono::steady_clock::now();
  std::cout << "CpuNaive " << (end - start).count() << " u-secs\n";

  for (int i = 0; i < 8; ++i) {
    std::cout << "\tSUM " << i << ": " << sums[i] << std::endl;
  }
}

void CpuNaiveWithSizes(const AlignedVector<Point4F> &u_data,
                       const AlignedVector<int> &u_sizes,
                       const AlignedVector<Point3F> &u_query,
                       const int num_batches, const int batch_size) {
  const size_t data_size = num_batches * batch_size;
  auto functor = MyFunctor();

  std::vector<Point3F> sums(num_batches);

  auto start = std::chrono::steady_clock::now();

  for (int tid = 0; tid < data_size; ++tid) {
    const auto batch_id = tid / batch_size;
    const auto idx_in_batch = tid % batch_size;
    const auto size_in_batch = u_sizes[batch_id];

    if (idx_in_batch < size_in_batch) {
      sums[batch_id] += functor(u_data[tid], u_query[batch_id]);
    }
  }

  auto end = std::chrono::steady_clock::now();
  std::cout << "CpuNaiveWithSizes " << (end - start).count() << " u-secs\n";

  for (int i = 0; i < 8; ++i) {
    std::cout << "\tSUM " << i << ": " << sums[i] << std::endl;
  }
}

void CpuBestWithSizes(const AlignedVector<Point4F> &u_data,
                      const AlignedVector<int> &u_sizes,
                      const AlignedVector<Point3F> &u_query,
                      const int num_batches, const int batch_size) {
  const size_t data_size = num_batches * batch_size;
  auto functor = MyFunctor();

  std::vector<Point3F> sums(num_batches);

  auto start = std::chrono::steady_clock::now();

  for (int batch = 0; batch < num_batches; ++batch) {
    const auto size_in_batch = u_sizes[batch];
    for (int i = 0; i < size_in_batch; ++i) {
      sums[batch] += functor(u_data[batch * batch_size + i], u_query[batch]);
    }
  }

  auto end = std::chrono::steady_clock::now();
  std::cout << "CpuBestWithSizes " << (end - start).count() << " u-secs\n";

  for (int i = 0; i < 8; ++i) {
    std::cout << "\tSUM " << i << ": " << sums[i] << std::endl;
  }
}

void ComputeTreeReductionWithSizes(sycl::queue &q,
                                   const AlignedVector<Point4F> &u_data,
                                   const AlignedVector<int> &u_sizes,
                                   const AlignedVector<Point3F> &u_query,
                                   const int num_batches,
                                   const int batch_size) {
  const size_t data_size = num_batches * batch_size;
  auto functor = MyFunctor();

  std::vector<Point3F> sums(num_batches);

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};

  // Each 4 work groups process a batch
  int work_group_size = 256;

  assert(batch_size % work_group_size == 0);

  // const int num_work_items = data_size;
  // const int num_work_groups = num_work_items / work_group_size;

  auto start = std::chrono::steady_clock::now();

  sycl::buffer<Point4F> data_buf(u_data.data(), data_size, props);
  // sycl::buffer<Point3F> query_buf(u_query.data(), num_batches, props);
  sycl::buffer<Point3F> accum_buf(num_batches);

  // const auto groups_per_batch =
  //     batch_size / work_group_size;  // 1024->4, 4k->16. etc

  // ComputeParallel2 main begin

  int batch_id = 0;
  const auto items_in_batch = (unsigned long)u_sizes[batch_id];
  const auto query_in_batch = u_query[batch_id];

  q.submit([&](auto &h) {
    sycl::accessor buf_acc(data_buf, h, sycl::read_only);
    sycl::accessor accum_acc(accum_buf, h, sycl::write_only, sycl::no_init);

    sycl::range num_items{items_in_batch};

    h.parallel_for(num_items, [=](auto idx) {
      accum_acc[batch_id] += functor(buf_acc[idx], query_in_batch);
    });
  });
  q.wait();
  {
    // sycl::host_accessor h_acc(accum_buf);
    // int group_id = 0;
    // for (int i = 0; i < num_batches; ++i) {
    //   for (int j = 0; j < groups_per_batch; ++j) {
    //     sums[i] += h_acc[group_id + j];
    //   }
    //   group_id += groups_per_batch;
    // }
  }

  auto end = std::chrono::steady_clock::now();
  std::cout << "ComputeTreeReduction1 " << (end - start).count() << " u-secs\n";

  for (int i = 0; i < 8; ++i) {
    std::cout << "\tSUM " << i << ": " << sums[i] << std::endl;
  }
}

void ComputeTreeReduction(sycl::queue &q, const AlignedVector<Point4F> &u_data,
                          const AlignedVector<Point3F> &u_query,
                          const int num_batches, const int batch_size) {
  const size_t data_size = num_batches * batch_size;
  auto functor = MyFunctor();

  std::vector<Point3F> sums(num_batches);

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};

  // Each 4 work groups process a batch
  int work_group_size = 256;

  assert(batch_size % work_group_size == 0);

  const int num_work_items = data_size;
  const int num_work_groups = num_work_items / work_group_size;
  // 1024*1024 / 256 = 4096 total work groups
  // So basically, each BH batch (size 1024) need 4 work groups
  // We can use 'num_work_groups / 4' to get which 'batch_id' it is

  // assert(batch_size / work_group_size == 4);

  int max_work_group_size =
      q.get_device().get_info<sycl::info::device::max_work_group_size>();
  if (work_group_size > max_work_group_size) {
    std::cout << "WARNING: Skipping one stage reduction example "
              << "as the device does not support required work_group_size"
              << std::endl;
    return;
  }

  std::cout << "num_work_items " << num_work_items << std::endl;
  std::cout << "num_work_groups " << num_work_groups << std::endl;
  std::cout << "batch_size / work_group_size " << batch_size / work_group_size
            << std::endl;

  auto start = std::chrono::steady_clock::now();

  sycl::buffer<Point4F> data_buf(u_data.data(), data_size, props);
  sycl::buffer<Point3F> query_buf(u_query.data(), num_batches, props);
  sycl::buffer<Point3F> accum_buf(num_work_groups);

  const auto groups_per_batch =
      batch_size / work_group_size;  // 1024->4, 4k->16. etc

  // ComputeParallel2 main begin
  q.submit([&](auto &h) {
    sycl::accessor buf_acc(data_buf, h, sycl::read_only);
    sycl::accessor query_acc(query_buf, h, sycl::read_only);
    sycl::accessor accum_acc(accum_buf, h, sycl::write_only, sycl::no_init);
    sycl::local_accessor<Point3F, 1> scratch(work_group_size, h);

    h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
                   [=](sycl::nd_item<1> item) {
                     size_t global_id = item.get_global_id(0);
                     int local_id = item.get_local_id(0);
                     int group_id = item.get_group(0);
                     const auto batch_id = group_id / groups_per_batch;

                     if (global_id < data_size) {
                       scratch[local_id] =
                           functor(buf_acc[global_id], query_acc[batch_id]);
                     } else
                       scratch[local_id] = Point3F{0.0f, 0.0f, 0.0f};

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
  // ComputeParallel2 main end
  q.wait();
  {
    sycl::host_accessor h_acc(accum_buf);
    int group_id = 0;
    for (int i = 0; i < num_batches; ++i) {
      for (int j = 0; j < groups_per_batch; ++j) {
        sums[i] += h_acc[group_id + j];
      }
      group_id += groups_per_batch;
    }
  }

  auto end = std::chrono::steady_clock::now();
  std::cout << "ComputeTreeReduction1 " << (end - start).count() << " u-secs\n";

  for (int i = 0; i < 8; ++i) {
    std::cout << "\tSUM " << i << ": " << sums[i] << std::endl;
  }
}

void ComputeSyclReduction(sycl::queue &q, const AlignedVector<float> &u_data) {
  const size_t data_size = u_data.size();

  float sum;

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};

  auto start = std::chrono::steady_clock::now();

  sycl::buffer<float> data_buf(u_data.data(), data_size, props);
  sycl::buffer<float> accum_buf(&sum, 1);
  sycl::range num_items{data_size};

  q.submit([&](auto &h) {
    sycl::accessor buf_acc(data_buf, h, sycl::read_only);
    sycl::accessor sum_acc(accum_buf, h, sycl::read_write);
    auto sumr = sycl::reduction(sum_acc, sycl::plus<>());
    h.parallel_for(sycl::nd_range<1>{data_size, 256}, sumr,
                   [=](sycl::nd_item<1> item, auto &sumr_arg) {
                     int glob_id = item.get_global_id(0);
                     sumr_arg += buf_acc[glob_id];
                   });
  });

  // q.submit([&](auto &h) {
  //   sycl::accessor buf_acc(data_buf, h, sycl::read_only);
  //   sycl::accessor accum_acc(accum_buf, h, sycl::read_write);

  //   h.parallel_for(num_items, sycl::reduction(accum_acc, sycl::plus<>()),
  //                  [=](auto idx, auto &sum) { sum += buf_acc[idx]; });
  // });
  q.wait();
  sycl::host_accessor h_acc(accum_buf);
  std::cout << "\tSUM: " << h_acc[0] << std::endl;

  auto end = std::chrono::steady_clock::now();
  std::cout << "ComputeSyclReduction " << (end - start).count() << " u-secs\n";
}

void WarmUp(sycl::queue &q) {
  int sum;
  sycl::buffer<int> sum_buf(&sum, 1);
  q.submit([&](auto &h) {
    sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);
    h.parallel_for(1, [=](auto) { sum_acc[0] = 0; });
  });
  q.wait();
}

struct Buffer {
  Buffer() {}

  // AlignedVector<Point4F> u_data;
  // AlignedVector<int> u_sizes;
  // AlignedVector<Point3F> u_queries;
};

int main() {
  sycl::queue q(d_selector);
  // WarmUp(q);
  // VectorAllocator<float> alloc(q);
  // AlignedVector<float> u_data(1024 * 1024, alloc);
  // ComputeSyclReduction(q, u_data);

  const auto num_batch = 1024;
  const auto batch_size = 4 * 1024;

  VectorAllocator<Point4F> p4f_alloc(q);
  VectorAllocator<Point3F> p3f_alloc(q);
  VectorAllocator<int> int_alloc(q);

  AlignedVector<Point4F> u_data(num_batch * batch_size, p4f_alloc);
  AlignedVector<int> u_sizes(num_batch, int_alloc);
  AlignedVector<Point3F> u_queries(num_batch, p3f_alloc);
  // AlignedVector<Point3F> u_results(num_batch, p3f_alloc);

  std::generate(u_data.begin(), u_data.end(), MakeRandomPoint<4, float>);
  std::generate(u_queries.begin(), u_queries.end(), MakeRandomPoint<3, float>);
  // std::fill(u_results.begin(), u_results.end(), Point3F());

  for (int i = 0; i < num_batch; ++i) {
    u_sizes[i] = (int)(my_rand(16.0f, 32.0f)) * 32;
  }

  // WarmUp(q);

  // CpuNaive(u_data, u_queries, num_batch, batch_size);
  // CpuNaiveWithSizes(u_data, u_sizes, u_queries, num_batch, batch_size);
  // // CpuBestWithSizes(u_data, u_sizes, u_queries, num_batch, batch_size);

  // ComputeTreeReduction(q, u_data, u_queries, num_batch, batch_size);
  // ComputeTreeReductionWithSizes(q, u_data, u_sizes, u_queries, num_batch,
  //                               batch_size);

  return 0;
}