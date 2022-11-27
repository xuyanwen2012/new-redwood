#include <CL/sycl.hpp>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../../src/Redwood.hpp"
#include "../../src/sycl/Utils.hpp"
#include "KnnExecutor.hpp"
#include "KnnSet.hpp"

struct NnBuffer;
struct KnnResult;

// ------------------- Application Types -------------------

using DataT = Point4F;
using QueryT = Point4F;
using ResultT = float;  // Special for KNN

// ------------------- Constants -------------------
constexpr auto kBlockThreads = 256;
constexpr auto kNumStreams = 2;  // Assume 2
constexpr auto kK = 32;          // Assume

constexpr auto kDebugPrint = true;

int stored_leaf_size;
int stored_num_threads;
int stored_num_batches;

sycl::default_selector d_selector;
sycl::device device;
sycl::context ctx;

// ------------------- Thread local -------------------

struct ReducerHandler {
  sycl::queue qs[kNumStreams];
  std::vector<NnBuffer> nn_buffers;
  std::vector<KnnResult> knn_results;
  int cur_collecting = 0;

  std::vector<ResultT> temp_sorter;  // stored_leaf_size + kK

  _NODISCARD NnBuffer& CurrentBuffer() { return nn_buffers[cur_collecting]; }
  _NODISCARD sycl::queue& CurrentQueue() { return qs[cur_collecting]; }
  _NODISCARD KnnResult& CurrentResults() { return knn_results[cur_collecting]; }
};

std::vector<ReducerHandler> rhs;

// ------------------- Global Shared  -------------------

const DataT* host_leaf_node_table_ref;
std::unique_ptr<sycl::buffer<DataT>> leaf_node_table_buf;

// ------------------- Stats -------------------
int leaf_reduced_counter = 0;
std::vector<int> num_branch_visited;
std::vector<int> num_leaf_visited;

// -------------- Buffer related -------------

struct KnnResult {
  KnnResult(const sycl::queue& q, const int num_query)
      : results_usm(num_query * kK, redwood::UsmAlloc<ResultT>(q)) {
    std::fill(results_usm.begin(), results_usm.end(),
              std::numeric_limits<ResultT>::max());
  }
  redwood::UsmVector<ResultT> results_usm;  // k * m
};

// Knn buffer is the same as Nn buffer (it is just indices);
struct NnBuffer {
  NnBuffer() = delete;

  NnBuffer(const sycl::queue& q, const int num_batch)
      : tasks(redwood::UsmAlloc<redwood::Task>(q)),
        leaf_idx(redwood::IntAlloc(q)) {
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

  redwood::UsmVector<redwood::Task> tasks;  // num_batch
  redwood::UsmVector<int> leaf_idx;         // num_batch
};

void ProcessKnnBuffer(sycl::queue& q, const NnBuffer& buffer, KnnResult& result,
                      const int num_batch_to_process) {
  constexpr auto kernel_func = MyFunctor();

  const auto task_ptr = buffer.tasks.data();
  const auto leaf_idx_ptr = buffer.leaf_idx.data();
  const auto result_ptr = result.results_usm.data();

  const auto local_leaf_size = stored_leaf_size;
  q.submit([&](sycl::handler& h) {
    const sycl::accessor leaf_table_acc(*leaf_node_table_buf, h,
                                        sycl::read_only);

    h.parallel_for(
        sycl::range(num_batch_to_process), [=](const sycl::id<1> idx) {
          const auto leaf_id = leaf_idx_ptr[idx];
          const auto query_point = task_ptr[idx].query_point;
          const auto query_idx = task_ptr[idx].query_idx;

          // Copy result K sets
          ResultT my_set[kK];
          for (int i = 0; i < kK; ++i)
            my_set[i] = result_ptr[query_idx * kK + i];

          for (int i = 0; i < local_leaf_size; ++i) {
            const auto dist = kernel_func(
                leaf_table_acc[leaf_id * local_leaf_size + i], query_point);

            //------------------- Lower Bound -----

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

            //------------------- Insert -----
            my_set[low] = dist;
          }

          for (int j = 0; j < kK; ++j)
            result_ptr[query_idx * kK + j] = my_set[j];
        });
  });
}

void ProcessKnnBufferCpu(const NnBuffer& buffer, KnnResult& result,
                         std::vector<float>& temp_sorter,
                         const int num_batch_to_process) {
  constexpr auto kernel_func = MyFunctor();

  auto result_ptr = result.results_usm.data();
  // static std::vector<ResultT> temp_sorter(stored_leaf_size + kK);

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

  ReducerHandler rh;
  rh.cur_collecting = 0;

  rh.qs[0] = sycl::queue(device);
  const auto ctx = rh.qs[0].get_context();

  for (int i = 1; i < kNumStreams; i++) rh.qs[i] = sycl::queue(ctx, device);
  for (auto& q : rh.qs) {
    rh.nn_buffers.emplace_back(q, stored_num_batches);
  }
  rh.temp_sorter.resize(stored_leaf_size + kK);
  rhs.push_back(rh);

  for (int i = 1; i < stored_num_threads; ++i) {
    ReducerHandler rh;
    rh.cur_collecting = 0;

    for (int i = 0; i < kNumStreams; i++) rh.qs[i] = sycl::queue(ctx, device);

    for (auto& q : rh.qs) {
      rh.nn_buffers.emplace_back(q, stored_num_batches);
    }
    rh.temp_sorter.resize(stored_leaf_size + kK);
    rhs.push_back(rh);
  }

  ShowDevice(rhs[0].qs[0]);
  WarmUp(rhs[0].qs[0]);
}

void ReduceLeafNode(const long tid, const unsigned node_idx,
                    const unsigned query_idx) {}

void GetReductionResult(const long tid, const unsigned query_idx,
                        void* result) {
  auto addr = static_cast<ResultT**>(result);
  // *addr = &knn_results[cur_collecting].results_usm[query_idx * kK];
  *addr = &rhs[tid].CurrentResults().results_usm[query_idx * kK];
}

void SetQueryPoints(const long tid, const void* query_points,
                    const unsigned num_query) {
  std::cout << "SetQueryPoints " << tid << std::endl;

  const auto total_count = num_query * kK;

  rhs[tid].knn_results.reserve(kNumStreams);
  rhs[tid].knn_results.emplace_back(rhs[tid].qs[0], num_query);
  rhs[tid].knn_results.emplace_back(rhs[tid].qs[1], num_query);
}

void SetNodeTables(const void* leaf_node_table, const unsigned num_leaf_nodes) {
  host_leaf_node_table_ref = static_cast<const DataT*>(leaf_node_table);
  leaf_node_table_buf = std::make_unique<sycl::buffer<DataT>>(
      host_leaf_node_table_ref, sycl::range{num_leaf_nodes * stored_leaf_size});
}

void ExecuteBatchedKernelsAsync(long tid, const int num_batch_collected) {
  // exit(1);
  constexpr auto threshold = 64;
  // ProcessKnnBufferCpu(rhs[tid].CurrentBuffer(), rhs[tid].CurrentResults(),
  //                     rhs[tid].temp_sorter, num_batch_collected);

  if (num_batch_collected < threshold) {
    ProcessKnnBufferCpu(rhs[tid].CurrentBuffer(), rhs[tid].CurrentResults(),
                        rhs[tid].temp_sorter, num_batch_collected);
  } else {
    if (tid == 0) {
      ProcessKnnBuffer(rhs[tid].CurrentQueue(), rhs[tid].CurrentBuffer(),
                       rhs[tid].CurrentResults(), num_batch_collected);
    } else {
      ProcessKnnBufferCpu(rhs[tid].CurrentBuffer(), rhs[tid].CurrentResults(),
                          rhs[tid].temp_sorter, num_batch_collected);
    }
  }

  // if constexpr (kDebugPrint) {
  //   std::cout << '[' << tid << ']' << ": cur, " << rhs[tid].cur_collecting
  //             << std::endl;
  // }

  const auto next = 1 - rhs[tid].cur_collecting;
  rhs[tid].qs[next].wait();

  rhs[tid].nn_buffers[next].Clear();
  rhs[tid].cur_collecting = next;
}

void ReduceLeafNodeWithTask(long tid, const unsigned node_idx,
                            const void* task) {
  const auto t = static_cast<const Task*>(task);
  // nn_buffers[cur_collecting].PushNewTask(*t, node_idx);
  rhs[tid].CurrentBuffer().PushNewTask(*t, node_idx);
}

void EndReducer() {}
}  // namespace redwood
