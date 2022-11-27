#include <CL/sycl.hpp>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../../src/Redwood.hpp"
#include "Executor.hpp"
#include "KnnSet.hpp"

struct NnBuffer;

template <typename T>
using UsmAlloc = sycl::usm_allocator<T, sycl::usm::alloc::shared>;

template <typename T>
using UsmVector = std::vector<T, UsmAlloc<T>>;

using IntAlloc = UsmAlloc<int>;
using FloatAlloc = UsmAlloc<float>;

// ------------------- Constants -------------------
constexpr auto kBlockThreads = 256;
constexpr auto kNumStreams = 2;  // Assume 2
constexpr auto kK = 32;          // Assume

int stored_leaf_size;
int stored_num_threads;
int stored_num_batches;

sycl::default_selector d_selector;
sycl::device device;
sycl::context ctx;

// ------------------- Vars -------------------

sycl::queue qs[kNumStreams];
std::vector<NnBuffer> nn_buffers;

std::vector<UsmVector<float>> knn_results_usm;  // k * m
std::vector<UsmVector<float>> temp_dist_usm;    // num_batch * (leaf_size)

int cur_collecting;

// ------------------- Shared Constant Pointer to  -------------------

// const Point4F* host_query_points_ref;

const Point4F* host_leaf_node_table_ref;
std::unique_ptr<sycl::buffer<Point4F>> leaf_node_table_buf;

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

// Knn buffer is the same as Nn buffer (it is just indices);
struct NnBuffer {
  NnBuffer() = delete;

  NnBuffer(const sycl::queue& q, const int num_batch)
      : tasks(UsmAlloc<redwood::Task>(q)), leaf_idx(IntAlloc(q)) {
    tasks.reserve(num_batch);
    leaf_idx.reserve(num_batch);
  }

  _NODISCARD size_t Size() const { return leaf_idx.size(); }

  void Clear() {
    // TODO: no need to clear every time, just overwrite the value
    tasks.clear();
    leaf_idx.clear();
  }

  void PushNewTask(const redwood::Task& task, const int leaf_id) {
    tasks.push_back(task);
    leaf_idx.push_back(leaf_id);
  }

  void UpdateLeafForExistingTask(const int batch_id, const int new_leaf_id) {
    // Make sure 'batch_id < this.Size()';
    leaf_idx[batch_id] = new_leaf_id;
  }

  UsmVector<redwood::Task> tasks;
  UsmVector<int> leaf_idx;
};

void ProcessNnBuffer(sycl::queue& q, const NnBuffer& buffer,
                     const int num_batch_to_process) {
  constexpr auto kernel_func = MyFunctor();

  const auto leaf_size = stored_leaf_size;

  const auto task_ptr = buffer.tasks.data();
  const auto leaf_idx_ptr = buffer.leaf_idx.data();
  auto result_ptr = knn_results_usm[cur_collecting].data();

  q.submit([&](sycl::handler& h) {
    const sycl::accessor leaf_table_acc(*leaf_node_table_buf, h,
                                        sycl::read_only);

    h.parallel_for(
        sycl::range(num_batch_to_process), [=](const sycl::id<1> idx) {
          const auto leaf_id = leaf_idx_ptr[idx];
          const auto query_point = task_ptr[idx].query_point;
          const auto query_idx = task_ptr[idx].query_idx;

          float my_set[kK];
          for (int i = 0; i < kK; ++i)
            my_set[i] = result_ptr[query_idx * kK + i];

          for (int i = 0; i < leaf_size; ++i) {
            const auto dist = kernel_func(
                leaf_table_acc[leaf_id * leaf_size + i], query_point);

            //------------------- Lower Bound -----
            // float dist = i / 10.0 + 0.00001f;

            int first = 0;
            int count = kK;
            while (count > 0) {
              int it = first;
              const int step = count / 2;
              it += step;
              if (my_set[it] < dist) {
                first = ++it;
                count -= step + 1;
              } else
                count = step;
            }

            //------------------- Right shift -----

            const auto low = first;
            for (int j = kK - 1; j > low; --j) {
              my_set[j] = my_set[j - 1];
            }

            //------------------- Right shift -----
            my_set[low] = dist;
          }

          for (int j = 0; j < kK; ++j)
            result_ptr[query_idx * kK + j] = my_set[j];
        });
  });
}

void ProcessNnBufferCpu(const NnBuffer& buffer,
                        const int num_batch_to_process) {
  constexpr auto kernel_func = MyFunctor();

  auto result_ptr = knn_results_usm[cur_collecting].data();
  static std::vector<float> temp_sorter(stored_leaf_size + kK);

  for (int batch_id = 0; batch_id < num_batch_to_process; ++batch_id) {
    const auto leaf_id = buffer.leaf_idx[batch_id];
    const auto query_point = buffer.tasks[batch_id].query_point;
    const auto query_idx = buffer.tasks[batch_id].query_idx;

    for (int i = 0; i < stored_leaf_size; ++i) {
      const auto dist =
          kernel_func(host_leaf_node_table_ref[leaf_id * stored_leaf_size + i],
                      query_point);

      temp_sorter[i] = dist;
    }
    std::copy_n(result_ptr + query_idx * kK, kK,
                temp_sorter.begin() + stored_leaf_size);

    std::nth_element(temp_sorter.begin(), temp_sorter.begin() + kK,
                     temp_sorter.end());
    std::copy_n(temp_sorter.begin(), kK, result_ptr + query_idx * kK);
  }
}

namespace redwood {
void InitReducer(const unsigned num_threads, const unsigned leaf_size,
                 const unsigned batch_num, const unsigned batch_size) {
  stored_num_threads = static_cast<int>(num_threads);
  stored_leaf_size = static_cast<int>(leaf_size);
  stored_num_batches = static_cast<int>(batch_num);

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
    nn_buffers.emplace_back(q, stored_num_batches);
  }

  WarmUp(qs[0]);
  cur_collecting = 0;

  // Temp
  const auto total_count = stored_num_batches * (leaf_size);
  temp_dist_usm.reserve(kNumStreams);
  temp_dist_usm.emplace_back(total_count, FloatAlloc(qs[0]));
  temp_dist_usm.emplace_back(total_count, FloatAlloc(qs[1]));
}

void StartQuery(const long tid, const unsigned query_idx) {}

void ReduceLeafNode(const long tid, const unsigned node_idx,
                    const unsigned query_idx) {}

void ReduceBranchNode(const long tid, const void* node_element,
                      unsigned query_idx) {}

void GetReductionResult(const long tid, const unsigned query_idx,
                        void* result) {
  auto addr = static_cast<float**>(result);
  *addr = knn_results_usm[cur_collecting].data() + query_idx * kK;
}

void SetQueryPoints(const long tid, const void* query_points,
                    const unsigned num_query) {
  const auto total_count = num_query * kK;
  knn_results_usm.reserve(kNumStreams);
  knn_results_usm.emplace_back(total_count, FloatAlloc(qs[0]));
  knn_results_usm.emplace_back(total_count, FloatAlloc(qs[1]));
  std::fill(knn_results_usm[0].begin(), knn_results_usm[0].end(),
            std::numeric_limits<float>::max());
  std::fill(knn_results_usm[1].begin(), knn_results_usm[1].end(),
            std::numeric_limits<float>::max());

  // std::make_shared<PersistentResults<float, kK>>(usm.data());
}

void SetNodeTables(const void* leaf_node_table, const unsigned num_leaf_nodes) {
  host_leaf_node_table_ref = static_cast<const Point4F*>(leaf_node_table);
  leaf_node_table_buf = std::make_unique<sycl::buffer<Point4F>>(
      host_leaf_node_table_ref, sycl::range{num_leaf_nodes * stored_leaf_size});
}

void SetNodeTables(const void* leaf_node_table, const unsigned* leaf_node_sizes,
                   const unsigned num_leaf_nodes) {
  SetNodeTables(leaf_node_table, num_leaf_nodes);
}

void ExecuteBatchedKernelsAsync(long tid, const int num_batch_collected) {
  constexpr auto threshold = 64;
  if (num_batch_collected < threshold) {
    ProcessNnBufferCpu(nn_buffers[cur_collecting], num_batch_collected);
  } else {
    // StartProcessBuffer(qs[cur_collecting], nn_buffers[cur_collecting],
    // num_batch_collected);

    ProcessNnBuffer(qs[cur_collecting], nn_buffers[cur_collecting],
                    num_batch_collected);
  }

  const auto next = 1 - cur_collecting;
  qs[next].wait();

  if (num_batch_collected >= threshold) {
    // FinishProcessBuffer(nn_buffers[next], next);
  }

  nn_buffers[next].Clear();
  cur_collecting = next;
}

void ReduceLeafNodeWithTask(long tid, const unsigned node_idx,
                            const void* task) {
  const auto t = static_cast<const Task*>(task);
  nn_buffers[cur_collecting].PushNewTask(*t, node_idx);
}

void* GetUnifiedResultLocationBase(long tid) {
  return knn_results_usm[cur_collecting].data();
}

void* GetUnifiedResultLocation(long tid, const int query_idx) {
  return knn_results_usm[cur_collecting].data() + query_idx * kK;
}

void EndReducer() {
  // const auto next = 1 - cur_collecting;
  // qs[next].wait();
}
}  // namespace redwood
