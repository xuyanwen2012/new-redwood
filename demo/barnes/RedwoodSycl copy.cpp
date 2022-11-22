#include <CL/sycl.hpp>

#include "../PointCloud.hpp"
#include "Kernel.cuh"

static auto exception_handler = [](sycl::exception_list eList) {
  for (std::exception_ptr const& e : eList) {
    try {
      std::rethrow_exception(e);
    } catch (std::exception const& e) {
      // #if DEBUG
      std::cout << "Failure" << std::endl;
      // #endif
      std::terminate();
    }
  }
};

namespace redwood {

// ------------------- Constants -------------------
constexpr auto kBlockThreads = 256;
constexpr auto kNumStreams = 1;
int stored_leaf_size;
int stored_num_threads;
int stored_num_batches;  // num_batches == num_blocks == num_executors
int stored_batch_size;   // better to be multiple of 'num_threads'

sycl::device device;
sycl::context ctx;
sycl::property_list props;

// -------------------  Buffer -------------------

struct BarnesBranchBatch {
  BarnesBranchBatch() = default;

  ~BarnesBranchBatch() {
    sycl::free(u_data, ctx);
    sycl::free(u_query_idx, ctx);
    sycl::free(u_items_in_batch, ctx);
  }

  // Each block takes a Batch, so 'num == num_blocks'
  void AllocateBuffer(const int num, const int size) {
    Reset();
    // stored_num_batches = num;
    // stored_batch_size = size;

    const auto bytes = sizeof(Point4F) * num * size;
    u_data =
        static_cast<Point4F*>(sycl::malloc_shared(bytes, device, ctx, props));

    const auto unsigned_bytes = sizeof(unsigned) * num;
    u_query_idx = static_cast<unsigned*>(
        sycl::malloc_shared(unsigned_bytes, device, ctx, props));

    u_items_in_batch = static_cast<unsigned*>(
        sycl::malloc_shared(unsigned_bytes, device, ctx, props));

    // std::cout << " ========= " << bytes << std::endl;
  }

  // Called when API "OnStartQuery()" is called
  void OnStartQuery(const unsigned query_idx) {
    if (current_batch != -1) {
      u_items_in_batch[current_batch] = current_idx_in_batch;
    }

    ++current_batch;
    u_query_idx[current_batch] = query_idx;
    current_idx_in_batch = 0;
  }

  // Called when API "ReduceBranchNode()" is called
  void LoadBranchNode(const unsigned q_idx, const Point4F* com) {
    // std::cout << "44444444444444444444: " << current_batch << ", "
    //           << current_idx_in_batch << std::endl;

    // // Batch overflow, use the next one
    // if (current_idx_in_batch == stored_batch_size) {
    //   u_items_in_batch[current_batch] = current_idx_in_batch;
    //   ++current_batch;

    //   u_query_idx[current_batch] = q_idx;
    //   current_idx_in_batch = 0;
    // }

    // std::cout << "555555555555555555555" << std::endl;

    u_data[current_batch * stored_batch_size + current_idx_in_batch] = *com;
    ++current_idx_in_batch;
  }

  void EndTraversal() {
    u_items_in_batch[current_batch] = current_idx_in_batch;
  }

  _NODISCARD unsigned Size(const unsigned batch_id) const {
    return u_items_in_batch[batch_id];
  }

  void Reset() {
    current_batch = -1;
    current_idx_in_batch = 0;
  }

  // Center of masses and query points
  unsigned* u_items_in_batch;  // n
  Point4F* u_data;             // n * size
  unsigned* u_query_idx;       // n

  // Misc
  int current_idx_in_batch;
  int current_batch;
};

// BarnesBranchBatch theOnlyBatch;

struct ReducerHandler {
  // TODO: Fix
  void Init() {
    streams[0] = sycl::queue(device, exception_handler);
    for (int i = 1; i < kNumStreams; i++) {
      streams[i] = sycl::queue(streams[0].get_context(), device);
    }
    cur_collecting = 0;
  }

  void AllocateBuffers() {
    for (int i = 0; i < kNumStreams; i++) {
      br_batches[i].AllocateBuffer(stored_num_batches, stored_batch_size);
    }
  }

  void AllocateResults(const unsigned m) {
    const auto bytes = sizeof(Point3F) * m;
    u_results =
        static_cast<Point3F*>(sycl::malloc_shared(bytes, device, ctx, props));

    std::fill_n(u_results, m, Point3F());
  }

  sycl::buffer<Point3F>* b_query_data;
  Point3F* u_results;

  unsigned cur_collecting;
  sycl::queue streams[kNumStreams];
  BarnesBranchBatch br_batches[kNumStreams];

  sycl::queue& CurrentStream() { return streams[cur_collecting]; }
  BarnesBranchBatch& CurrentBranchBatch() { return br_batches[cur_collecting]; }
};

// num_threads
std::vector<ReducerHandler> rhs;

// -------------------  SYCL Related -------------------

class MyReduction;

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

  // This avoids the need to copy the content of the buffer back and forth
  // between the host memory and the buffer memory, potentially saving time
  // during buffer creation and destruction.
  props = {sycl::property::buffer::use_host_ptr()};
}

// -------------------  Application Related -------------------

// Shared Data
sycl::buffer<Point4F>* b_node_content_table;

namespace internal {

void AllocateAndCopyLeafNodesData(const void* leaf_node_table,
                                  const unsigned num_leaf_nodes) {
  auto lnd = static_cast<const Point4F*>(leaf_node_table);
  b_node_content_table = new sycl::buffer(
      lnd, sycl::range{num_leaf_nodes * stored_leaf_size}, props);
}

void AllocateAndCopyQueryData(const long tid, const void* query_data,
                              const unsigned m) {
  auto qd = static_cast<const Point3F*>(query_data);
  rhs[tid].b_query_data = new sycl::buffer(qd, sycl::range{m}, props);
}

}  // namespace internal

static inline int RoundUpIntDiv(int x, int y) { return (x + y - 1) / y; }

// For Branch
void ExecuteKernelBranch(const long tid, BarnesBranchBatch& batch) {
  batch.EndTraversal();

  const auto num_collected = batch.current_batch;  // something like 1024
  const auto iterations = RoundUpIntDiv(num_collected, kBlockThreads);

  std::cout << num_collected << ", " << iterations << std::endl;

  // Done on GPU
  rhs[tid].CurrentStream().submit([&](sycl::handler& cgh) {
    sycl::accessor acc_qd{*rhs[tid].b_query_data, cgh, sycl::read_only};

    const auto functor = MyFunctor();

    // Create Local Ref
    const Point4F* local_u_batch_data = batch.u_data;
    const unsigned* local_u_query_idx = batch.u_query_idx;
    const unsigned* local_u_items_in_batch = batch.u_items_in_batch;
    Point3F* local_u_results = rhs[tid].u_results;
    const int local_stored_batch_size = stored_batch_size;

    cgh.parallel_for<MyReduction>(
        sycl::range{kBlockThreads}, [=](sycl::id<1> tid) {
          for (int block = 0; block < iterations; ++block) {
            // This is the 'n' in 'batch[n]'
            const auto data_idx = block * kBlockThreads + tid;

            auto my_dist = Point3F();

            if (data_idx < num_collected) {
              const auto query_idx = local_u_query_idx[data_idx];
              const auto q = acc_qd[query_idx];
              const auto num_items = local_u_items_in_batch[query_idx];

              for (int i = 0; i < num_items; ++i) {
                const auto dist = functor(
                    local_u_batch_data[data_idx * local_stored_batch_size + i],
                    q);
                my_dist += dist;
              }

              local_u_results[query_idx] += my_dist;
            }
          }
        });
  });

  rhs[tid]
      .u_results

          rhs[tid]
      .CurrentStream()
      .wait();

  std::cout << "--------------" << std::endl;
  exit(1);
}

BarnesBranchBatch theOnlyBatch;

// -------------- REDwood APIs -------------
void InitReducer(const unsigned num_threads, const unsigned leaf_size,
                 const unsigned batch_num, const unsigned batch_size) {
  stored_num_threads = num_threads;
  stored_leaf_size = leaf_size;

  stored_num_batches = batch_num;
  stored_batch_size = batch_size;

  InitSycl();

  // theOnlyBatch.AllocateBuffer(batch_num, batch_size);

  rhs.resize(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    rhs[i].Init();
    rhs[i].AllocateBuffers();
  }
}

void StartQuery(const long tid, const unsigned query_idx) {
  // theOnlyBatch.OnStartQuery(query_idx);
  rhs[tid].CurrentBranchBatch().OnStartQuery(query_idx);
}

void ReduceLeafNode(const long tid, const unsigned node_idx,
                    const unsigned query_idx) {}

void ReduceBranchNode(const long tid, const void* node_element,
                      unsigned query_idx) {
  // std::cout << "query_idx" << std::endl;
  // theOnlyBatch.LoadBranchNode(query_idx,
  //                             static_cast<const Point4F*>(node_element));
  // std::cout << "2222222222222" << std::endl;

  rhs[tid].CurrentBranchBatch().LoadBranchNode(
      query_idx, static_cast<const Point4F*>(node_element));
}

void GetReductionResult(const long tid, const unsigned query_idx,
                        void* result) {
  // TODO: Add offset
  auto addr = static_cast<Point3F*>(result);
  *addr = rhs[tid].u_results[query_idx];
}

void SetQueryPoints(const long tid, const void* query_points,
                    const unsigned num_query) {
  internal::AllocateAndCopyQueryData(tid, query_points, num_query);
  rhs[tid].AllocateResults(num_query);
}

void SetNodeTables(const void* leaf_node_table, const unsigned num_leaf_nodes) {
  internal::AllocateAndCopyLeafNodesData(leaf_node_table, num_leaf_nodes);
}

void SetNodeTables(const void* leaf_node_table,
                   const unsigned* leaf_node_sizes_,
                   const unsigned num_leaf_nodes) {
  SetNodeTables(leaf_node_table, num_leaf_nodes);
}

void ExecuteBatchedKernelsAsync(long tid) {
  // std::cout << "11111111111" << std::endl;

  ExecuteKernelBranch(tid, rhs[tid].CurrentBranchBatch());

  // for (int i = 0; i < 4; ++i) {
  //   std::cout << i << ": " << rhs[tid].CurrentBranchBatch().u_query_idx[i]
  //             << "\t" << rhs[tid].CurrentBranchBatch().u_items_in_batch[i]
  //             << std::endl;
  //   for (int j = 0; j < 3; ++j) {
  //     std::cout
  //         << "\t" << j << ": "
  //         << rhs[tid].CurrentBranchBatch().u_data[i * stored_batch_size + j]
  //         << std::endl;
  //   }
  // }

  auto next_collecting = (rhs[tid].cur_collecting + 1) % kNumStreams;
  rhs[tid].streams[next_collecting].wait();

  rhs[tid].cur_collecting = next_collecting;
  rhs[tid].CurrentBranchBatch().Reset();

  // theOnlyBatch.EndTraversal();
  // theOnlyBatch.Reset();

  // for (int i = 0; i < 4; ++i) {
  //   std::cout << i << ": " << theOnlyBatch.u_query_idx[i] << "\t"
  //             << theOnlyBatch.u_items_in_batch[i] << std::endl;
  //   for (int j = 0; j < 3; ++j) {
  //     std::cout << "\t" << j << ": "
  //               << theOnlyBatch.u_data[i * stored_batch_size + j] <<
  //               std::endl;
  //   }
  //   std::cout << "\t..." << std::endl;
  //   for (int j = stored_batch_size - 3; j < stored_batch_size; ++j) {
  //     std::cout << "\t" << j << ": "
  //               << theOnlyBatch.u_data[i * stored_batch_size + j] <<
  //               std::endl;
  //   }
  // }

  // exit(0);
}

void EndReducer() {
  for (int i = 0; i < stored_num_threads; ++i) {
    rhs[i].CurrentStream().wait();
  }
}

}  // namespace redwood