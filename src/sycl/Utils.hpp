#pragma once

#include <CL/sycl.hpp>
#include <iomanip>
#include <iostream>

namespace redwood {

template <typename T>
using UsmAlloc = sycl::usm_allocator<T, sycl::usm::alloc::shared>;

template <typename T>
using UsmVector = std::vector<T, UsmAlloc<T>>;

using IntAlloc = UsmAlloc<int>;
using FloatAlloc = UsmAlloc<float>;

void ShowDevice(sycl::queue& q) {
  //   // Output platform and device information.
  //   auto device = q.get_device();
  //   auto p_name =
  //   device.get_platform().get_info<sycl::info::platform::name>(); std::cout
  //   << std::setw(20) << "Platform Name: " << p_name << "\n"; auto p_version =
  //       device.get_platform().get_info<sycl::info::platform::version>();
  //   std::cout << std::setw(20) << "Platform Version: " << p_version << "\n";
  //   auto d_name = device.get_info<sycl::info::device::name>();
  //   std::cout << std::setw(20) << "Device Name: " << d_name << "\n";
  //   auto max_work_group =
  //       device.get_info<sycl::info::device::max_work_group_size>();
  //   std::cout << std::setw(20) << "Max Work Group: " << max_work_group <<
  //   "\n"; auto max_compute_units =
  //       device.get_info<sycl::info::device::max_compute_units>();
  //   std::cout << std::setw(20) << "Max Compute Units: " << max_compute_units
  //             << "\n\n";
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

}  // namespace redwood
