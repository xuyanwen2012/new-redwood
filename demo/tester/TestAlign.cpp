//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <numeric>
#include <sycl/sycl.hpp>

#include "../PointCloud.hpp"

sycl::default_selector d_selector;

template <typename T>
using VectorAllocator = sycl::usm_allocator<T, sycl::usm::alloc::shared>;

template <typename T>
using AlignedVector = std::vector<T, VectorAllocator<T>>;

constexpr size_t array_size = (1 << 15);

// Snippet2 Begin
int VectorAdd1(sycl::queue &q, const AlignedVector<Point3F> &a,
               const AlignedVector<Point3F> &b, AlignedVector<Point3F> &sum,
               int iter) {
  sycl::range num_items{a.size()};

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};

  for (int i = 0; i < iter; i++) {
    sycl::buffer a_buf(a, props);
    sycl::buffer b_buf(b, props);
    sycl::buffer sum_buf(sum.data(), num_items, props);
    {
      sycl::host_accessor a_host_acc(a_buf);
      std::cout << "add1: buff memory address =" << a_host_acc.get_pointer()
                << "\n";
      std::cout << "add1: address of vector aa = " << a.data() << "\n";
    }
    q.submit([&](auto &h) {
      // Input accessors
      sycl::accessor a_acc(a_buf, h, sycl::read_only);
      sycl::accessor b_acc(b_buf, h, sycl::read_only);
      // Output accessor
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);
      sycl::stream out(16 * 1024, 16 * 1024, h);

      h.parallel_for(num_items, [=](auto i) {
        if (i[0] == 0)
          out << "add1: dev addr = " << a_acc.get_pointer() << "\n";
        sum_acc[i] = a_acc[i] + b_acc[i];
      });
    });
  }
  q.wait();
  return (0);
}
// Snippet2 End

// Snippet2 Begin
int NewTest(sycl::queue &q, const Point3F *a, const Point3F *b, Point3F *sum,
            const unsigned long size, int iter) {
  sycl::range num_items{size};

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};

  for (int i = 0; i < iter; i++) {
    sycl::buffer a_buf(a, num_items, props);
    sycl::buffer b_buf(b, num_items, props);
    sycl::buffer sum_buf(sum, num_items, props);
    {
      sycl::host_accessor a_host_acc(a_buf);
      std::cout << "add1: buff memory address =" << a_host_acc.get_pointer()
                << "\n";
      std::cout << "add1: address of vector aa = " << a << "\n";
    }
    q.submit([&](auto &h) {
      // Input accessors
      sycl::accessor a_acc(a_buf, h, sycl::read_only);
      sycl::accessor b_acc(b_buf, h, sycl::read_only);
      // Output accessor
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);
      sycl::stream out(16 * 1024, 16 * 1024, h);

      h.parallel_for(num_items, [=](auto i) {
        if (i[0] == 0)
          out << "add1: dev addr = " << a_acc.get_pointer() << "\n";
        sum_acc[i] = a_acc[i] + b_acc[i];
      });
    });
  }
  q.wait();
  return (0);
}
// Snippet2 End

// Snippet3 Begin
int VectorAdd2(sycl::queue &q, AlignedVector<Point3F> &a,
               AlignedVector<Point3F> &b, AlignedVector<Point3F> &sum,
               int iter) {
  sycl::range num_items{a.size()};

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iter; i++) {
    sycl::buffer a_buf(a, props);
    sycl::buffer b_buf(b, props);
    sycl::buffer sum_buf(sum.data(), num_items, props);
    q.submit([&](auto &h) {
      // Input accessors
      sycl::accessor a_acc(a_buf, h, sycl::read_only);
      sycl::accessor b_acc(b_buf, h, sycl::read_only);
      // Output accessor
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(num_items,
                     [=](auto i) { sum_acc[i] = a_acc[i] + b_acc[i]; });
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  std::cout << "Vector add2 completed on device - took "
            << (end - start).count() << " u-secs\n";
  return ((end - start).count());
}
// Snippet3 End

int VectorAdd4(sycl::queue &q, const std::vector<Point3F> &a,
               const std::vector<Point3F> &b, std::vector<Point3F> &sum,
               int iter) {
  sycl::range num_items{a.size()};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iter; i++) {
    sycl::buffer a_buf(a);
    sycl::buffer b_buf(b);
    sycl::buffer sum_buf(sum.data(), num_items);
    auto e = q.submit([&](auto &h) {
      // Input accessors
      sycl::accessor a_acc(a_buf, h, sycl::read_only);
      sycl::accessor b_acc(b_buf, h, sycl::read_only);
      // Output accessor
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(num_items,
                     [=](auto i) { sum_acc[i] = a_acc[i] + b_acc[i]; });
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  std::cout << "Vector add4 completed on device - took "
            << (end - start).count() << " u-secs\n";
  return ((end - start).count());
}

int main() {
  sycl::queue q(d_selector);
  auto device = q.get_device();
  sycl::context ctx;

  Point3F *a;
  Point3F *b;
  Point3F *c;
  a = sycl::aligned_alloc_shared<Point3F>(64, sizeof(Point3F) * array_size,
                                          device, ctx);
  b = sycl::aligned_alloc_shared<Point3F>(64, sizeof(Point3F) * array_size,
                                          device, ctx);
  c = sycl::aligned_alloc_shared<Point3F>(64, sizeof(Point3F) * array_size,
                                          device, ctx);

  for (int i = 0; i < array_size; ++i) {
    a[i] = Point3F{(float)i, 0.0f, 0.0f};
    b[i] = Point3F{(float)i, 0.0f, 0.0f};
    c[i] = Point3F();
  }

  NewTest(q, a, b, c, array_size, 1);

  for (int i = 0; i < 16; i++) {
    std::cout << c[i] << std::endl;
  }

  // VectorAllocator<Point3F> alloc(q);
  // AlignedVector<Point3F> a(array_size, alloc);
  // AlignedVector<Point3F> b(array_size, alloc);
  // AlignedVector<Point3F> c(array_size, alloc);

  // std::iota(a.begin(), a.end(), Point3F());
  // std::iota(b.begin(), b.end(), Point3F());

  // std::cout << "Running on device: "
  //           << q.get_device().get_info<sycl::info::device::name>() << "\n";
  // std::cout << "Vector size: " << a.size() << "\n";

  // // jit the code
  // VectorAdd1(q, a, b, c, 1);

  // std::vector<Point3F> h_a(array_size);
  // std::vector<Point3F> h_b(array_size);
  // std::vector<Point3F> h_c(array_size);
  // std::iota(h_a.begin(), h_a.end(), Point3F());
  // std::iota(h_b.begin(), h_b.end(), Point3F());
  // VectorAdd4(q, h_a, h_b, h_c, 1000);

  // VectorAdd2(q, a, b, c, 1000);

  // for (int i = 0; i < 16; i++) {
  //   std::cout << c[i] << std::endl;
  // }

  return 0;
}