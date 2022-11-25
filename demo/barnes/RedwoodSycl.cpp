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
constexpr auto kOffloadBranchNode = false;

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

std::unique_ptr<sycl::buffer<DataT>> leaf_node_table_buf;
std::unique_ptr<sycl::buffer<unsigned>> leaf_sizes_buf;

// I think these should be put in to the buffer struct
// int leaf_reduced_counter = 0;

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
      : my_task(),
        leaf_nodes(IntAlloc(q)),
        branch_data(UsmAlloc<DataT>(q)),
        tmp_results_br(UsmAlloc<ResultT>(q)),
        tmp_count_br(),
        tmp_results_le(UsmAlloc<ResultT>(q)),
        tmp_count_le() {
    // If it grow larger then let it happen. I don't care
    leaf_nodes.reserve(batch_size);
    branch_data.reserve(batch_size);

    // For this one, as long as your traversal did not get more than 256 * 1024
    // branch nodes, then you are fine
    tmp_results_le.resize(1024);
    if constexpr (kOffloadBranchNode) tmp_results_br.resize(1024);
  }

  _NODISCARD size_t NumLeafsCollected() const { return leaf_nodes.size(); }
  _NODISCARD size_t NumBranchCollected() const { return branch_data.size(); }

  void Clear() {
    // No need to clear the 'tem_result_br's, they will be overwrite
    tmp_count_br = 0;
    tmp_count_le = 0;
    leaf_nodes.clear();
    branch_data.clear();
  }

  void SetTask(const redwood::Task<QueryT> task) { my_task = task; }

  void PushLeaf(const int leaf_id) { leaf_nodes.push_back(leaf_id); }
  void PushBranch(const DataT& com) { branch_data.push_back(com); }

  // Actual batch data , a single task with many many branch/leaf_idx
  redwood::Task<QueryT> my_task;
  UsmVector<int> leaf_nodes;
  UsmVector<DataT> branch_data;

  // Some temporary space used for itermediate results,
  UsmVector<ResultT> tmp_results_br;
  int tmp_count_br;
  UsmVector<ResultT> tmp_results_le;
  int tmp_count_le;
};

// Naive
void StartProcessBhLeafPack(sycl::queue& q, BhPack& pack) {
  constexpr auto kernel_func = MyFunctor();
  const auto data_size = pack.NumLeafsCollected();

  if (data_size == 0) {
    return;
  }

  const auto leaf_idx_ptr = pack.leaf_nodes.data();
  const auto query_point = pack.my_task.query_point;

  // Each work is a leaf, 'data_size' == leaf collected in the pack
  const auto num_work_items =
      MyRoundUp(data_size, static_cast<size_t>(kBlockThreads));
  const auto num_work_groups = num_work_items / kBlockThreads;

  // std::cout << "One Stage Reduction with " << num_work_items << " leafs. "
  //	<< "num_work_groups: " << num_work_groups << std::endl;

  if (num_work_groups > 1024) {
    std::cout << "should not happen" << std::endl;
    exit(1);
  }

  // DEBUG:
  // std::cout << data_size << std::endl;
  // for (int i = 0; i < data_size; ++i) {
  //   leaf_reduced_counter += host_leaf_sizes_ref[leaf_idx_ptr[i]];
  // }

  pack.tmp_count_le = num_work_groups;
  const auto tmp_result_ptr = pack.tmp_results_le.data();

  const auto max_leaf_size = stored_leaf_size;
  q.submit([&](sycl::handler& h) {
    const sycl::accessor leaf_table_acc(*leaf_node_table_buf, h,
                                        sycl::read_only);
    const sycl::accessor leaf_sizes_acc(*leaf_sizes_buf, h, sycl::read_only);

    const sycl::local_accessor<ResultT, 1> scratch(kBlockThreads, h);

    h.parallel_for(sycl::nd_range<1>(num_work_items, kBlockThreads),
                   [=](const sycl::nd_item<1> item) {
                     const auto global_id = item.get_global_id(0);
                     const auto local_id = item.get_local_id(0);
                     const auto group_id = item.get_group(0);

                     if (global_id < data_size) {
                       const auto leaf_id = leaf_idx_ptr[global_id];
                       const auto leaf_size = leaf_sizes_acc[leaf_id];

                       ResultT my_sum{};
                       for (int i = 0; i < leaf_size; ++i) {
                         my_sum += kernel_func(
                             leaf_table_acc[leaf_id * max_leaf_size + i],
                             query_point);
                       }
                       scratch[local_id] = my_sum;
                     } else
                       scratch[local_id] = ResultT();

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

void FinishProcessBhLeafPack(const BhPack& pack) {
  const auto count = pack.tmp_count_le;
  auto local = ResultT();
  for (int i = 0; i < count; ++i) {
    local += pack.tmp_results_le[i];
  }
  final_results[pack.my_task.query_idx] += local;
}

void StartProcessBhBranchPack(sycl::queue& q, BhPack& pack) {
  constexpr auto kernel_func = MyFunctor();
  const auto data_size = pack.NumBranchCollected();

  if (data_size == 0) {
    return;
  }

  const auto query_point = pack.my_task.query_point;
  const auto com_ptr = pack.branch_data.data();

  // Each work is a center of mass, 'data_size' == items to reduce
  const auto num_work_items =
      MyRoundUp(data_size, static_cast<size_t>(kBlockThreads));
  const auto num_work_groups = num_work_items / kBlockThreads;

  // std::cout << "One Stage Reduction with " << num_work_items << " leafs. "
  //	<< "num_work_groups: " << num_work_groups << std::endl;

  if (num_work_groups > 1024) {
    std::cout << "should not happen" << std::endl;
    exit(1);
  }

  pack.tmp_count_br = num_work_groups;
  const auto tmp_result_ptr = pack.tmp_results_br.data();

  q.submit([&](sycl::handler& h) {
    const sycl::local_accessor<ResultT, 1> scratch(kBlockThreads, h);

    h.parallel_for(sycl::nd_range<1>(num_work_items, kBlockThreads),
                   [=](const sycl::nd_item<1> item) {
                     const auto global_id = item.get_global_id(0);
                     const auto local_id = item.get_local_id(0);
                     const auto group_id = item.get_group(0);

                     if (global_id < data_size) {
                       scratch[local_id] =
                           kernel_func(com_ptr[global_id], query_point);
                     } else
                       scratch[local_id] = ResultT();

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

void FinishProcessBhBranchPack(const BhPack& pack) {
  const auto count = pack.tmp_count_br;
  auto local = ResultT();
  for (int i = 0; i < count; ++i) {
    local += pack.tmp_results_br[i];
  }
  final_results[pack.my_task.query_idx] += local;
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
  const auto task = static_cast<const Task<QueryT>*>(task_obj);
  bh_buffers[cur_collecting].SetTask(*task);
}

void ReduceLeafNode(const long tid, const unsigned node_idx,
                    const unsigned query_idx) {
  bh_buffers[cur_collecting].PushLeaf(node_idx);

  ++num_leaf_visited[query_idx];
}

void ReduceBranchNode(const long tid, const void* node_element,
                      const unsigned query_idx) {
  const auto com = static_cast<const DataT*>(node_element);

  if constexpr (kOffloadBranchNode) {
    bh_buffers[cur_collecting].PushBranch(*com);

  } else {
    constexpr auto functor = MyFunctor();
    final_results[query_idx] +=
        functor(*com, host_tasks_ref[query_idx].query_point);
  }

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

  num_branch_visited.resize(num_query);
  num_leaf_visited.resize(num_query);
}

void SetNodeTables(const void* leaf_node_table, const unsigned* leaf_node_sizes,
                   const unsigned num_leaf_nodes) {
  host_leaf_table_ref = static_cast<const DataT*>(leaf_node_table);
  leaf_node_table_buf = std::make_unique<sycl::buffer<DataT>>(
      host_leaf_table_ref, sycl::range{num_leaf_nodes * stored_leaf_size});

  host_leaf_sizes_ref = leaf_node_sizes;
  leaf_sizes_buf = std::make_unique<sycl::buffer<unsigned>>(
      host_leaf_sizes_ref, sycl::range{num_leaf_nodes});
}

void ExecuteBatchedKernelsAsync(long tid, const int num_batch_collected) {
  StartProcessBhLeafPack(qs[cur_collecting], bh_buffers[cur_collecting]);
  if constexpr (kOffloadBranchNode)
    StartProcessBhBranchPack(qs[cur_collecting], bh_buffers[cur_collecting]);

  const auto next = 1 - cur_collecting;
  qs[next].wait();

  FinishProcessBhLeafPack(bh_buffers[next]);
  if constexpr (kOffloadBranchNode) FinishProcessBhBranchPack(bh_buffers[next]);

  bh_buffers[next].Clear();
  cur_collecting = next;
}

void EndReducer() {
  if constexpr (false) {
    for (int i = 0; i < 32; ++i) {
      std::cout << i << ":\tbr: " << num_branch_visited[i]
                << "\tle: " << num_leaf_visited[i] << std::endl;
    }
  }

  // std::cout << "leaf_reduced_counter: " << leaf_reduced_counter << std::endl;

  const auto br_max =
      *std::max_element(num_branch_visited.begin(), num_branch_visited.end());
  const auto le_max =
      *std::max_element(num_leaf_visited.begin(), num_leaf_visited.end());

  std::cout << "Br Max: " << br_max << std::endl;
  std::cout << "Le Max: " << le_max << std::endl;
}
}  // namespace redwood
