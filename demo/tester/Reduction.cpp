#include <CL/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cstdlib>

#include "../../src/Utils.hpp"
#include "../PointCloud.hpp"
#include "../barnes/Kernel.cuh"

// ------------------- Constants -------------------
constexpr auto kBlockThreads = 256;
int stored_leaf_size;
int stored_num_threads;
int stored_num_batches;  // num_batches == num_blocks == num_executors
int stored_batch_size;   // better to be multiple of 'num_threads'

sycl::device device;
sycl::context ctx;
sycl::property_list props;

class Timer {
 public:
  Timer() : start_(std::chrono::steady_clock::now()) {}

  double Elapsed() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(now - start_).count();
  }

 private:
  using Duration = std::chrono::duration<double>;
  std::chrono::steady_clock::time_point start_;
};

struct DebugFunctor {
  Point3F operator()(const Point4F p, const Point3F q) const {
    return Point3F{p.data[0], p.data[1], p.data[2]};
  }
};

void ComputeSerial(const std::vector<Point4F> &h_in_data,
                   const Point3F u_q_data) {
  const size_t data_size = h_in_data.size();
  auto functor = MyFunctor();

  auto sum = Point3F();

  TimeTask("ComputeSerial", [&] {
    for (int i = 0; i < data_size; ++i) {
      sum += functor(h_in_data[i], u_q_data);
    }
  });

  std::cout << "ComputeSerial Result: " << sum << std::endl;
}

void ComputeParallel1(sycl::queue &q, const std::vector<Point4F> &h_in_data,
                      const Point3F u_q_data) {
  const size_t data_size = h_in_data.size();
  auto functor = MyFunctor();

  int num_processing_elements =
      q.get_device().get_info<sycl::info::device::max_compute_units>();
  // 24
  int BATCH =
      (data_size + num_processing_elements - 1) / num_processing_elements;
  std::cout << "BATCH = " << BATCH << std::endl;

  sycl::buffer<Point4F> buf(h_in_data.data(), data_size, props);
  sycl::buffer<Point3F> sum_buf(num_processing_elements);

  auto sum = Point3F();

  TimeTask("ComputeParallel1", [&] {
    q.submit([&](auto &h) {
      sycl::accessor buf_acc(buf, h, sycl::read_only);
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(data_size, [=](sycl::id<1> tid) {
        size_t start = tid * BATCH;
        size_t end = (tid + 1) * BATCH;
        if (end > data_size) end = data_size;
        auto sum = Point3F();
        for (size_t i = start; i < end; ++i) {
          sum += functor(buf_acc[i], u_q_data);
        }
        sum_acc[tid] = sum;
      });
      // ComputeParallel1 main end
    });
    q.wait();

    sycl::host_accessor h_acc(sum_buf);
    for (int i = 0; i < num_processing_elements; ++i) {
      sum += h_acc[i];
    }
  });

  std::cout << "ComputeParallel1 Result: " << sum << std::endl;
}

void ComputeTreeReduction1(sycl::queue &q,
                           const std::vector<Point4F> &h_in_data,
                           const Point3F u_q_data) {
  const size_t data_size = h_in_data.size();
  auto functor = MyFunctor();

  int work_group_size = 256;
  int num_work_items = data_size;
  int num_work_groups = num_work_items / work_group_size;

  int max_work_group_size =
      q.get_device().get_info<sycl::info::device::max_work_group_size>();
  if (work_group_size > max_work_group_size) {
    std::cout << "WARNING: Skipping one stage reduction example "
              << "as the device does not support required work_group_size"
              << std::endl;
    return;
  }

  std::cout << "One Stage Reduction with " << num_work_items << std::endl;

  sycl::buffer<Point4F> buf(h_in_data.data(), data_size, props);
  sycl::buffer<Point3F> accum_buf(num_work_groups);

  auto sum = Point3F();

  Timer timer;

  // TimeTask("ComputeTreeReduction1", [&] {
  auto evt = q.submit([&](auto &h) {
    sycl::accessor buf_acc(buf, h, sycl::read_only);
    sycl::accessor accum_acc(accum_buf, h, sycl::write_only, sycl::no_init);
    sycl::local_accessor<Point3F, 1> scratch(work_group_size, h);

    h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
                   [=](sycl::nd_item<1> item) {
                     size_t global_id = item.get_global_id(0);
                     int local_id = item.get_local_id(0);
                     int group_id = item.get_group(0);

                     if (global_id < data_size)
                       scratch[local_id] =
                           functor(buf_acc[global_id], u_q_data);
                     else
                       scratch[local_id] = Point3F();

                     // Do a tree reduction on items in work-group
                     for (int i = work_group_size / 2; i > 0; i >>= 1) {
                       item.barrier(sycl::access::fence_space::local_space);
                       if (local_id < i)
                         scratch[local_id] += scratch[local_id + i];
                     }

                     if (local_id == 0) accum_acc[group_id] = scratch[0];
                   });
  });

  double t1 = timer.Elapsed();

  q.wait();

  sycl::host_accessor h_acc(accum_buf);
  for (int i = 0; i < num_work_groups; ++i) sum += h_acc[i];
  // });

  double t2 = timer.Elapsed();
  auto startK =
      evt.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto endK =
      evt.get_profiling_info<sycl::info::event_profiling::command_end>();
  std::cout << "Kernel submission time: " << t1 << "secs\n";
  std::cout << "Kernel submission + execution time: " << t2 << "secs\n";
  std::cout << "Kernel execution time: "
            << ((double)(endK - startK)) / 1000000.0 << "secs\n";

  std::cout << "ComputeTreeReduction1 Result: " << sum << std::endl;
}

void ComputeSyclReduction(sycl::queue &q, const std::vector<Point4F> &h_in_data,
                          const Point3F u_q_data) {
  const size_t data_size = h_in_data.size();
  auto functor = MyFunctor();

  int work_group_size = 256;
  int num_work_groups = data_size / work_group_size;

  std::cout << "ComputeSyclReduction::num_work_groups: " << num_work_groups
            << std::endl;

  sycl::buffer<Point4F> buf(h_in_data.data(), data_size, props);
  sycl::buffer<Point3F> accum_buf(num_work_groups);

  auto sum = Point3F();

  Timer timer;

  TimeTask("ComputeSyclReduction", [&] {
    auto evt = q.submit([&](auto &h) {
      sycl::accessor buf_acc(buf, h, sycl::read_only);
      sycl::accessor accum_acc(accum_buf, h, sycl::write_only, sycl::no_init);
      sycl::local_accessor<Point3F, 1> scratch(work_group_size, h);

      h.parallel_for(sycl::nd_range<1>(data_size, work_group_size),
                     [=](sycl::nd_item<1> item) {
                       size_t global_id = item.get_global_id(0);
                       int local_id = item.get_local_id(0);
                       int group_id = item.get_group(0);

                       if (global_id < data_size)
                         scratch[local_id] =
                             functor(buf_acc[global_id], u_q_data);
                       else
                         scratch[local_id] = Point3F();

                       // Do a tree reduction on items in work-group
                       for (int i = work_group_size / 2; i > 0; i >>= 1) {
                         item.barrier(sycl::access::fence_space::local_space);
                         if (local_id < i)
                           scratch[local_id] += scratch[local_id + i];
                       }

                       if (local_id == 0) accum_acc[group_id] = scratch[0];
                     });
    });

    double t1 = timer.Elapsed();

    q.wait();

    sycl::host_accessor h_acc(accum_buf);
    for (int i = 0; i < num_work_groups; ++i) sum += h_acc[i];
  });

  std::cout << "ComputeSyclReduction Result: " << sum << std::endl;
}

void Warpup(sycl::queue &q) {
  int sum = 0;
  sycl::buffer<int> sum_buf(&sum, 1, props);

  q.submit([&](auto &h) {
    sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

    h.parallel_for(1, [=](auto) { sum_acc[0] = 0; });
  });
  q.wait();
}

void InitSycl() {
  // Intel(R) UHD Graphics [0x9b41] on Parakeet
  device = sycl::device::get_devices(sycl::info::device_type::all)[1];

  std::cout << "SYCL Device: " << device.get_info<sycl::info::device::name>()
            << std::endl;

  int num_processing_elements =
      device.get_info<sycl::info::device::max_compute_units>();
  std::cout << "\t- Num work items = " << num_processing_elements << std::endl;

  int max_work_group_size =
      device.get_info<sycl::info::device::max_work_group_size>();
  std::cout << "\t- Max work group size = " << max_work_group_size << std::endl;

  int batch =
      (4096 * 1024 + num_processing_elements - 1) / num_processing_elements;
  std::cout << "\t- batch = " << num_processing_elements << std::endl;

  int vec_size = device.get_info<sycl::info::device::native_vector_width_int>();
  std::cout << "\t- Vec Size = " << vec_size << std::endl;

  props = {sycl::property::buffer::use_host_ptr()};
}

int main() {
  constexpr auto n = 8 * 1024;
  std::vector<Point4F> h_in_data(n);
  std::generate(h_in_data.begin(), h_in_data.end(), MakeRandomPoint<4, float>);
  // auto counter = 0.0f;
  // std::generate(h_in_data.begin(), h_in_data.end(), [&] {
  // return Point4F{counter++, 1.0f, 0.0f, 0.0f};
  // });

  const Point3F query_point{0.5f, 0.5f, 0.5f};
  InitSycl();

  ComputeSerial(h_in_data, query_point);

  // exception_handler
  sycl::queue q =
      sycl::queue(device, sycl::property::queue::enable_profiling());

  Warpup(q);

  ComputeParallel1(q, h_in_data, query_point);

  ComputeTreeReduction1(q, h_in_data, query_point);

  ComputeSyclReduction(q, h_in_data, query_point);

  return EXIT_SUCCESS;
}