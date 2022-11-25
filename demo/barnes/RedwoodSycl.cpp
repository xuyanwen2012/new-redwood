#include <CL/sycl.hpp>
#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "../../src/Redwood.hpp"
#include "../PointCloud.hpp"
#include "Executor.hpp"
#include "Kernel.hpp"

struct BhPack;

template <typename T>
using UsmAlloc = sycl::usm_allocator<T, sycl::usm::alloc::shared>;

template <typename T>
using UsmVector = std::vector<T, UsmAlloc<T>>;

using IntAlloc = UsmAlloc<int>;
using FloatAlloc = UsmAlloc<float>;

// ------------------- Application Types -------------------

using DataT = Point4F;
using QueryT = Point3F;
using ResultT = Point3F;

// ------------------- Constants -------------------
constexpr auto kBlockThreads = 256;
constexpr auto kNumStreams = 2;  // Assume 2
constexpr auto kK = 32;          // Assume

int stored_leaf_size;
int stored_num_threads;
int stored_num_batches;
int stored_batch_size;

sycl::default_selector d_selector;
sycl::device device;
sycl::context ctx;

// ------------------- Vars -------------------

sycl::queue qs[kNumStreams];
std::vector<BhPack> bh_buffers;

int cur_collecting;

// ------------------- Shared Constant Pointer to  -------------------

// Shared
const redwood::Task<QueryT>* host_tasks_ref;
const DataT* host_leaf_table_ref;
const unsigned* host_leaf_sizes_ref;
std::vector<ResultT> final_results;

std::unique_ptr<sycl::buffer<Point4F>> leaf_node_table_buf;
std::unique_ptr<sycl::buffer<unsigned>> leaf_sizes_buf;

std::vector<UsmVector<Point3F>> tmp_results;
std::array<int, kNumStreams> tmp_count;

// Stats
std::vector<int> num_branch_visited;
std::vector<int> num_leaf_visited;

// -------------- SYCL related -------------

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

// -------------- Buffer related -------------

// Single query
struct BhPack {
  BhPack() = delete;

  BhPack(const sycl::queue& q, const int batch_size)
      : my_task(), leaf_nodes(IntAlloc(q)) {
    // If it grow larger then let it happen. I don't care
    leaf_nodes.reserve(batch_size);
  }

  _NODISCARD size_t Size() const { return leaf_nodes.size(); }

  void Clear() { leaf_nodes.clear(); }

  void SetTask(const redwood::Task<Point3F> task) { my_task = task; }

  void PushLeaf(const int leaf_id) { leaf_nodes.push_back(leaf_id); }

  redwood::Task<Point3F> my_task;
  UsmVector<int> leaf_nodes;
};

// Naive
void StartProcessBhPack(sycl::queue& q, const BhPack& pack) {
  constexpr auto kernel_func = MyFunctor();
  const auto data_size = pack.Size();

  if (data_size == 0) {
    return;
  }

  const auto leaf_idx_ptr = pack.leaf_nodes.data();
  const auto query_point = pack.my_task.query_point;
  const auto query_idx = pack.my_task.query_idx;
  const auto leaf_size = stored_leaf_size;

  // Each work is a leaf, 'data_size' == leaf collected in the pack
  const auto num_work_items =
      MyRoundUp(data_size, static_cast<size_t>(kBlockThreads));
  const auto num_work_groups = num_work_items / kBlockThreads;

  // std::cout << "One Stage Reduction with " << num_work_items << " leafs. "
  //	<< "num_work_groups: " << num_work_groups << std::endl;

  auto sum = Point3F();

  if (num_work_groups > 1024) {
    std::cout << "should not happen" << std::endl;
    exit(1);
  }
  tmp_count[cur_collecting] = num_work_groups;
  const auto tmp_result_ptr = tmp_results[cur_collecting].data();

  q.submit([&](sycl::handler& h) {
    const sycl::accessor leaf_table_acc(*leaf_node_table_buf, h,
                                        sycl::read_only);
    const sycl::accessor leaf_sizes_acc(*leaf_sizes_buf, h, sycl::read_only);

    // const sycl::accessor<Point3F, 1, sycl::access::mode::read_write,
    //                      sycl::access::target::local>
    //     scratch(kBlockThreads, h);
    const sycl::local_accessor<Point3F, 1> scratch(kBlockThreads, h);

    h.parallel_for(sycl::nd_range<1>(num_work_items, kBlockThreads),
                   [=](const sycl::nd_item<1> item) {
                     const auto global_id = item.get_global_id(0);
                     const auto local_id = item.get_local_id(0);
                     const auto group_id = item.get_group(0);

                     if (global_id < data_size) {
                       const auto leaf_id = leaf_idx_ptr[global_id];
                       const auto leaf_size = leaf_sizes_acc[leaf_id];

                       Point3F my_sum{};
                       for (int i = 0; i < leaf_size; ++i) {
                         my_sum += kernel_func(
                             leaf_table_acc[leaf_id * leaf_size + i],
                             query_point);
                       }
                       scratch[local_id] = my_sum;
                     } else
                       scratch[local_id] = Point3F();

                     // Do a tree reduction on items in work-group
                     for (int i = kBlockThreads / 2; i > 0; i >>= 1) {
                       item.barrier(sycl::access::fence_space::local_space);
                       if (local_id < i)
                         scratch[local_id] += scratch[local_id + i];
                     }

                     if (local_id == 0) tmp_result_ptr[group_id] = scratch[0];
                   });
  });
}

void FinishProcessBhPack(const BhPack& pack, const int my_cur) {
  const auto count = tmp_count[my_cur];
  auto local = Point3F();
  for (int i = 0; i < count; ++i) {
    local += tmp_results[my_cur][i];
  }
  final_results[pack.my_task.query_idx] += local;

  // std::cout << final_results[pack.my_task.query_idx] << std::endl;
  tmp_count[my_cur] = 0;
}

namespace redwood {

void InitReducer(const unsigned num_threads, const unsigned leaf_size,
                 const unsigned batch_num, const unsigned batch_size) {
  stored_num_threads = static_cast<int>(num_threads);
  stored_leaf_size = static_cast<int>(leaf_size);
  stored_num_batches = static_cast<int>(batch_num);
  stored_batch_size = static_cast<int>(batch_size);

  try {
    device = sycl::device(sycl::gpu_selector());
  } catch (const sycl::exception& e) {
    std::cout << "Cannot select a GPU\n" << e.what() << "\n";
    exit(1);
  }

  qs[0] = sycl::queue(device);
  for (int i = 1; i < kNumStreams; i++)
    qs[i] = sycl::queue(qs[0].get_context(), device);

  ShowDevice(qs[0]);

  for (auto& q : qs) {
    bh_buffers.emplace_back(q, batch_size);
  }

  WarmUp(qs[0]);
  cur_collecting = 0;
}

void StartQuery(long tid, const void* task_obj) {
  const auto task = static_cast<const Task<Point3F>*>(task_obj);
  bh_buffers[cur_collecting].SetTask(*task);
}

void ReduceLeafNode(const long tid, const unsigned node_idx,
                    const unsigned query_idx) {
  bh_buffers[cur_collecting].PushLeaf(node_idx);

  ++num_leaf_visited[query_idx];
}

void ReduceBranchNode(const long tid, const void* node_element,
                      const unsigned query_idx) {
  // return;
  constexpr auto functor = MyFunctor();
  auto com = static_cast<const DataT*>(node_element);

  final_results[query_idx] +=
      functor(*com, host_tasks_ref[query_idx].query_point);

  ++num_branch_visited[query_idx];
}

void GetReductionResult(const long tid, const unsigned query_idx,
                        void* result) {
  const auto addr = static_cast<ResultT*>(result);
  *addr = final_results[query_idx];
}

void SetQueryPoints(long tid, const void* query_points, unsigned num_query) {
  host_tasks_ref = static_cast<const Task<QueryT>*>(query_points);
  final_results.resize(num_query);

  tmp_results.reserve(kNumStreams);
  tmp_results.emplace_back(1024, UsmAlloc<Point3F>(qs[0]));
  tmp_results.emplace_back(1024, UsmAlloc<Point3F>(qs[1]));

  num_branch_visited.resize(num_query);
  num_leaf_visited.resize(num_query);
}

void SetNodeTables(const void* leaf_node_table, const unsigned* leaf_node_sizes,
                   const unsigned num_leaf_nodes) {
  host_leaf_table_ref = static_cast<const DataT*>(leaf_node_table);
  leaf_node_table_buf = std::make_unique<sycl::buffer<Point4F>>(
      host_leaf_table_ref, sycl::range{num_leaf_nodes * stored_leaf_size});

  host_leaf_sizes_ref = leaf_node_sizes;
  leaf_sizes_buf = std::make_unique<sycl::buffer<unsigned>>(
      host_leaf_sizes_ref, sycl::range{num_leaf_nodes});
}

void ExecuteBatchedKernelsAsync(long tid, const int num_batch_collected) {
  StartProcessBhPack(qs[cur_collecting], bh_buffers[cur_collecting]);

  const auto next = 1 - cur_collecting;
  qs[next].wait();

  FinishProcessBhPack(bh_buffers[next], next);

  bh_buffers[next].Clear();
  cur_collecting = next;
}

void EndReducer() {
  if constexpr (false) {
    for (int i = 0; i < 256; ++i) {
      std::cout << i << ":\tbr: " << num_branch_visited[i]
                << "\tle: " << num_leaf_visited[i] << std::endl;
    }
  }

  const auto br_max =
      *std::max_element(num_branch_visited.begin(), num_branch_visited.end());
  const auto le_max =
      *std::max_element(num_leaf_visited.begin(), num_leaf_visited.end());

  std::cout << "Br Max: " << br_max << std::endl;
  std::cout << "Le Max: " << le_max << std::endl;
}

}  // namespace redwood