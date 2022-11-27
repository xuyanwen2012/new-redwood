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

void ShowDevice(const sycl::queue& q) {
  // Output platform and device information.
  const auto device = q.get_device();
  const auto p_name =
      device.get_platform().get_info<sycl::info::platform::name>();
  std::cout << std::setw(20) << "Platform Name: " << p_name << "\n";
  const auto p_version =
      device.get_platform().get_info<sycl::info::platform::version>();
  std::cout << std::setw(20) << "Platform Version: " << p_version << "\n";
  const auto d_name = device.get_info<sycl::info::device::name>();
  std::cout << std::setw(20) << "Device Name: " << d_name << "\n";
  const auto max_work_group =
      device.get_info<sycl::info::device::max_work_group_size>();
  std::cout << std::setw(20) << "Max Work Group: " << max_work_group << "\n";
  const auto max_compute_units =
      device.get_info<sycl::info::device::max_compute_units>();
  std::cout << std::setw(20) << "Max Compute Units: " << max_compute_units
            << "\n\n";
}

static auto exception_handler = [](sycl::exception_list eList) {
  for (const std::exception_ptr& e : eList) {
    try {
      std::rethrow_exception(e);
    } catch (const std::exception& e) {
#if DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

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
