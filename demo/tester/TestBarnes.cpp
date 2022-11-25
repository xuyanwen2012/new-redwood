#include <CL/sycl.hpp>
#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "../../src/Utils.hpp"
#include "../PointCloud.hpp"

struct NnBuffer;
template <typename T>
using UsmAlloc = sycl::usm_allocator<T, sycl::usm::alloc::shared>;

template <typename T>
using UsmVector = std::vector<T, UsmAlloc<T>>;

using IntAlloc = UsmAlloc<int>;
using FloatAlloc = UsmAlloc<float>;

sycl::default_selector d_selector;

struct MyFunctor {
  // GPU version
  float operator()(const Point4F p, const Point4F q) const {
    auto dist = float();

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const auto diff = p.data[i] - q.data[i];
      dist += diff * diff;
    }

    return sqrtf(dist);
  }
};

struct BarnesFunctor {
  static auto rsqrtf(const float x) { return 1.0f / sqrtf(x); }

  // GPU version
  _REDWOOD_KERNEL Point3F operator()(const Point4F p, const Point3F q) const {
    // For SYCL backend, use negative points to indicate invalid
    // if (p.data[0] < 0.0f) return Point3F();

    const auto dx = p.data[0] - q.data[0];
    const auto dy = p.data[1] - q.data[1];
    const auto dz = p.data[2] - q.data[2];
    const auto dist_sqr = dx * dx + dy * dy + dz * dz + 1e-9f;
    const auto inv_dist = rsqrtf(dist_sqr);
    const auto inv_dist3 = inv_dist * inv_dist * inv_dist;
    const auto with_mass = inv_dist3 * p.data[3];
    return {dx * with_mass, dy * with_mass, dz * with_mass};
  }
};

void ShowDevice(sycl::queue& q) {
  // Output platform and device information.
  auto device = q.get_device();
  auto p_name = device.get_platform().get_info<sycl::info::platform::name>();
  std::cout << std::setw(20) << "Platform Name: " << p_name << "\n";
  auto p_version =
      device.get_platform().get_info<sycl::info::platform::version>();
  std::cout << std::setw(20) << "Platform Version: " << p_version << "\n";
  auto d_name = device.get_info<sycl::info::device::name>();
  std::cout << std::setw(20) << "Device Name: " << d_name << "\n";
  auto max_work_group =
      device.get_info<sycl::info::device::max_work_group_size>();
  std::cout << std::setw(20) << "Max Work Group: " << max_work_group << "\n";
  auto max_compute_units =
      device.get_info<sycl::info::device::max_compute_units>();
  std::cout << std::setw(20) << "Max Compute Units: " << max_compute_units
            << "\n\n";
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

struct BarnesBranchBuffer {
  BarnesBranchBuffer(const sycl::queue& q, const int buffer_size) {
    data = std::make_unique<UsmVector<Point4F>>(UsmAlloc<Point4F>(q));
    data->reserve(buffer_size);
  }

  void SetQuery(const Point3F q_point, const int q_idx) {
    query_point = q_point;
    query_idx = q_idx;
  }

  void Push(const Point4F& com) const { data->push_back(com); }

  void Clear() const { data->clear(); }

  _NODISCARD size_t Size() const { return data->size(); }

  int query_idx;
  Point3F query_point;
  std::unique_ptr<UsmVector<Point4F>> data;
  std::shared_ptr<sycl::buffer<Point3F>> all_result_buf_ptr;
};

struct NnBuffer {
  NnBuffer(const sycl::queue& q, const int buffer_size, const int m) {
    query_idx = std::make_unique<UsmVector<int>>(buffer_size, IntAlloc(q));
    leaf_idx = std::make_unique<UsmVector<int>>(buffer_size, IntAlloc(q));
    results = std::make_unique<UsmVector<float>>(m, FloatAlloc(q));
  }

  _NODISCARD size_t Size() const { return query_idx->size(); }

  void InitRandomData() {
    std::iota(query_idx->begin(), query_idx->end(), 0);
    std::generate(leaf_idx->begin(), leaf_idx->end(),
                  [] { return static_cast<int>(my_rand(0.0, 1024.0)); });
    std::fill(results->begin(), results->end(),
              std::numeric_limits<float>::max());
  }

  void FakeTraversal() {
    std::iota(query_idx->begin(), query_idx->end(), 0);
    std::generate(leaf_idx->begin(), leaf_idx->end(),
                  [] { return static_cast<int>(my_rand(0.0, 1024.0)); });
  }

  std::unique_ptr<UsmVector<int>> query_idx;  // buffer_size
  std::unique_ptr<UsmVector<int>> leaf_idx;   // buffer_size
  std::unique_ptr<UsmVector<float>> results;  // m for now
};

void StartProcessBufferNaive(sycl::queue& q,
                             sycl::buffer<Point4F>& leaf_node_table_buf,
                             sycl::buffer<Point4F>& queries_table_buf,
                             NnBuffer& buffer, const int leaf_size) {
  constexpr auto kernel_func = MyFunctor();

  const auto query_idx_ptr = buffer.query_idx->data();
  const auto leaf_idx_ptr = buffer.leaf_idx->data();
  const auto results_ptr = buffer.results->data();
  q.submit([&](sycl::handler& h) {
    const sycl::accessor leaf_table_acc(leaf_node_table_buf, h,
                                        sycl::read_only);
    const sycl::accessor query_table_acc(queries_table_buf, h, sycl::read_only);
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
}

Point3F my_total_sum;
void ProcessBarnesBranch(sycl::queue& q, const BarnesBranchBuffer& barnes) {
  constexpr auto kernel_func = BarnesFunctor();
  const auto data_size = barnes.Size();

  constexpr auto work_group_size = 256;
  const auto num_work_items = MyRoundUp<size_t>(data_size, work_group_size);
  const auto num_work_groups = num_work_items / work_group_size;
  //   std::cout << "One Stage Reduction with " << num_work_items << std::endl;

  const auto query_idx = barnes.query_idx;
  const auto query_point = barnes.query_point;
  const auto data_ptr = barnes.data->data();

  Point3F sum{};
  sycl::buffer<Point3F> accum_buf(num_work_groups);
  q.submit([&](auto& h) {
    sycl::accessor accum_acc(accum_buf, h, sycl::write_only, sycl::no_init);
    sycl::local_accessor<Point3F, 1> scratch(work_group_size, h);

    h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
                   [=](const sycl::nd_item<1> item) {
                     const auto global_id = item.get_global_id(0);
                     const auto local_id = item.get_local_id(0);
                     const auto group_id = item.get_group(0);

                     if (global_id < data_size)
                       scratch[local_id] =
                           kernel_func(data_ptr[global_id], query_point);
                     else
                       scratch[local_id] = Point3F();

                     // Do a tree reduction on items in work-group
                     for (auto i = work_group_size / 2; i > 0; i >>= 1) {
                       item.barrier(sycl::access::fence_space::local_space);
                       if (local_id < i)
                         scratch[local_id] += scratch[local_id + i];
                     }

                     if (local_id == 0) accum_acc[group_id] = scratch[0];
                   });
  });
  // ComputeTreeReduction1 main end
  //   q.wait();
  {
    // sycl::host_accessor h_acc(accum_buf);
    // for (int i = 0; i < num_work_groups; ++i) sum += h_acc[i];
  }
  my_total_sum += sum;
  // std::cout << "sum " << sum << std::endl;
}

int main(int argc, char* argv[]) {
  constexpr auto kNumStreams = 2;
  const auto m = 2 * 1024;
  const auto num_leaf_nodes = 1024;
  const auto leaf_size = 256;
  const auto buffer_size = 2 * 1024;

  sycl::queue qs[kNumStreams];
  auto device = sycl::device::get_devices(sycl::info::device_type::all)[1];

  qs[0] = sycl::queue{device};
  qs[1] = {qs[0].get_context(), device};

  ShowDevice(qs[0]);

  TimeTask("Warpup 0", [&] { WarmUp(qs[0]); });
  TimeTask("Warpup 1", [&] { WarmUp(qs[1]); });

  std::array<BarnesBranchBuffer, kNumStreams> barnes_buffer{
      BarnesBranchBuffer(qs[0], 4 * 1024),
      BarnesBranchBuffer(qs[1], 4 * 1024),
  };

  std::vector<Point3F> q_points(m);
  std::generate(q_points.begin(), q_points.end(), MakeRandomPoint<3, float>);

  for (int i = 0; i < 4000; ++i) {
    barnes_buffer[0].Push(MakeRandomPoint<4, float>());
  }

  std::cout << "Query " << barnes_buffer[0].query_idx
            << " Collected: " << barnes_buffer[0].Size() << std::endl;

  Point3F total_sum{};
  TimeTask("CPU", [&] {
    constexpr auto functor = BarnesFunctor();
    for (int q_idx = 0; q_idx < m; ++q_idx) {
      barnes_buffer[0].SetQuery(q_points[q_idx], q_idx);
      Point3F sum{};
      for (int i = 0; i < barnes_buffer[0].Size(); ++i) {
        sum += functor(barnes_buffer[0].data->at(i), q_points[q_idx]);
      }
      //   barnes_buffer[0].Clear();
      // std::cout << "CPU sum: " << sum << std::endl;
      total_sum += sum;
    }
  });

  std::cout << "CPU sum: " << total_sum << std::endl;

  my_total_sum = Point3F();
  TimeTask("Naive", [&] {
    for (int q_idx = 0; q_idx < m; ++q_idx) {
      barnes_buffer[0].SetQuery(q_points[q_idx], q_idx);
      ProcessBarnesBranch(qs[0], barnes_buffer[0]);
      //   barnes_buffer[0].Clear();
    }
  });

  std::cout << "My sum: " << my_total_sum << std::endl;

  return 0;
}
