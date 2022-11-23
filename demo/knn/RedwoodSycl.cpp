#include <CL/sycl.hpp>
#include <algorithm>
#include <memory>
#include <vector>

#include "../PointCloud.hpp"
#include "Kernel.cuh"
#include "KnnResultSet.hpp"

namespace redwood {

// Vector types
struct alignas(8) uint2 {
  unsigned int x, y;
};

static inline uint2 make_uint2(unsigned int x, unsigned int y) {
  uint2 t;
  t.x = x;
  t.y = y;
  return t;
}

template <typename T>
using VectorAllocator = sycl::usm_allocator<T, sycl::usm::alloc::shared>;

template <typename T>
using AlignedVector = std::vector<T, VectorAllocator<T>>;

// ------------------- Constants -------------------
constexpr auto kBlockThreads = 256;
constexpr auto kNumStreams = 1;

constexpr auto kK = 32;
int stored_leaf_size;
int stored_num_threads;
int stored_num_batches;  // num_batches == num_work_groups == num_executors

sycl::device device;
sycl::context ctx;
sycl::property_list props;

// ------------------- Variables -------------------

sycl::queue q[kNumStreams];
int cur_collecting;

// ------------------- Shared Data -------------------

const Point4F* q_points_ref;
std::unique_ptr<sycl::buffer<Point4F>> b_query_point_buf;

const Point4F* lnd_ref;
std::unique_ptr<sycl::buffer<Point4F>>
    b_node_data_buf;  // Constant, so this buffer will always be there

// if 'm' queryies, then 'm * k'
std::vector<float> le_results;
std::vector<KnnResultSet<float, int>> result_sets;

// std::unique_ptr<sycl::buffer<Point3F>> b_result_buf;

// -------------- Buffer related -------------

struct NnBatch {
  NnBatch(const int num) {
    next_avalible_slot = 0;
    stored_num_batches = num;

    const auto bytes = sizeof(uint2) * num;
    u_buffer =
        static_cast<uint2*>(sycl::malloc_shared(bytes, device, ctx, props));
  };

  ~NnBatch() { sycl::free(u_buffer, ctx); }

  void LoadLeafNode(const unsigned q_idx, const unsigned node_idx) {
    u_buffer[next_avalible_slot] = make_uint2(q_idx, node_idx);
    ++next_avalible_slot;
  }

  void Reset() {
    // std::fill(u_buffer + next_avalible_slot, u_buffer + stored_num_batches,
    //           uint2{std::numeric_limits<unsigned>::max(),
    //                 std::numeric_limits<unsigned>::max()});

    next_avalible_slot = 0;
  }

  // the first element is 'query_index', and the second element is 'node_index'
  uint2* u_buffer;

  // Misc
  int next_avalible_slot;
  int stored_num_batches;
};

std::vector<NnBatch> batches;
static inline NnBatch& CurrentBatch() { return batches[cur_collecting]; }

// -------------- SYCL related -------------

void WarmUp(sycl::queue& q) {
  int sum;
  sycl::buffer<int> sum_buf(&sum, 1);
  q.submit([&](auto& h) {
    sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);
    h.parallel_for(1, [=](auto) { sum_acc[0] = 0; });
  });
  q.wait();
}

// -------------- REDwood APIs -------------
void InitReducer(const unsigned num_threads, const unsigned leaf_size,
                 const unsigned batch_num, const unsigned batch_size) {
  // Save Parameters
  stored_num_threads = num_threads;
  stored_leaf_size = leaf_size;
  stored_num_batches = batch_num;

  // Set up Sycl
  // Intel(R) UHD Graphics [0x9b41] on Parakeet
  device = sycl::device::get_devices(sycl::info::device_type::all)[1];
  std::cout << "SYCL Device: " << device.get_info<sycl::info::device::name>()
            << std::endl;
  props = {sycl::property::buffer::use_host_ptr()};

  q[0] = sycl::queue(device);
  for (int i = 1; i < kNumStreams; i++)
    q[i] = sycl::queue(q[0].get_context(), device);

  // Setup
  for (int i = 0; i < kNumStreams; i++) batches.emplace_back(batch_num);

  for (int i = 0; i < kNumStreams; i++) WarmUp(q[0]);

  cur_collecting = 0;
}

void StartQuery(const long tid, const unsigned query_idx) {}

void ReduceLeafNode(const long tid, const unsigned node_idx,
                    const unsigned query_idx) {
  batches[cur_collecting].LoadLeafNode(query_idx, node_idx);
}

void ReduceBranchNode(const long tid, const void* node_element,
                      unsigned query_idx) {
  constexpr auto kernel_func = MyFunctor();
  const auto p = static_cast<const Point4F*>(node_element);
  auto dist = kernel_func(*p, q_points_ref[query_idx]);
  result_sets[query_idx].AddPoint(dist);
}

void GetReductionResult(const long tid, const unsigned query_idx,
                        void* result) {
  // TODO: Add offset
  auto addr = static_cast<float**>(result);
  *addr = le_results.data() + query_idx * kK;
}

void SetQueryPoints(const long tid, const void* query_points,
                    const unsigned num_query) {
  q_points_ref = static_cast<const Point4F*>(query_points);
  b_query_point_buf = std::make_unique<sycl::buffer<Point4F>>(
      q_points_ref, sycl::range{num_query}, props);

  // Set results
  le_results.resize(num_query * kK);
  result_sets.reserve(num_query);
  for (int i = 0; i < num_query; ++i) {
    result_sets.emplace_back(kK);
    result_sets[i].Init(le_results.data() + i * kK);
  }
}

void SetNodeTables(const void* leaf_node_table, const unsigned num_leaf_nodes) {
  auto lnd = static_cast<const Point4F*>(leaf_node_table);
  lnd_ref = lnd;
  b_node_data_buf = std::make_unique<sycl::buffer<Point4F>>(
      lnd, sycl::range{num_leaf_nodes * stored_leaf_size}, props);
}

void SetNodeTables(const void* leaf_node_table,
                   const unsigned* leaf_node_sizes_,
                   const unsigned num_leaf_nodes) {
  SetNodeTables(leaf_node_table, num_leaf_nodes);
}

void ProcessBatch(NnBatch& batch) {
  constexpr auto functor = MyFunctor();

  for (int i = 0; i < stored_num_batches; ++i) {
    uint2 content = batches[cur_collecting].u_buffer[i];
    const auto q_idx = content.x;
    const auto q_point = q_points_ref[q_idx];
    const auto leaf_idx = content.y;

    for (int j = 0; j < stored_leaf_size; ++j) {
      const auto dist =
          functor(lnd_ref[leaf_idx * stored_leaf_size + j], q_point);
      result_sets[q_idx].AddPoint(dist);
    }
  }
}

void ProcessBatchSycl(sycl::queue& q, NnBatch& batch) {
  constexpr auto functor = MyFunctor();

  const auto total_num_items =
      stored_num_batches * stored_leaf_size;  // 1024 * 256

  // 1k->4, 256->1, For Sycl, lets have leaf size >256
  const auto items_per_thread = stored_leaf_size / kBlockThreads;

  // For 'nd_range'
  const auto num_total_works = total_num_items;  /// items_per_thread;

  // Potentially move this out
  std::vector<float> dists(total_num_items);

  {
    sycl::buffer dists_buf(dists);
    sycl::buffer<uint2> buffer_buf(batch.u_buffer, stored_num_batches);

    const auto local_stored_leaf_size = stored_leaf_size;

    q.submit([&](auto& h) {
      sycl::accessor leaf_table_acc(*b_node_data_buf, h, sycl::read_only);
      sycl::accessor query_table_acc(*b_query_point_buf, h, sycl::read_only);

      sycl::accessor buffer_acc(buffer_buf, h, sycl::read_only);

      sycl::accessor dist_acc(dists_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(
          sycl::nd_range<1>(num_total_works, kBlockThreads),
          [=](const sycl::nd_item<1> item) {
            const auto global_id = item.get_global_id(0);
            const auto local_id = item.get_local_id(0);  // 0...255
            const auto group_id = item.get_group(0);  // 0...1024 aka batch_id

            //const auto batch_id = ;
            //const auto index_in_batch = ;

            const uint2 content = buffer_acc[group_id];
            const auto q_idx = content.x;
            const auto leaf_idx = content.y;
            const auto q_point = query_table_acc[q_idx];

            dist_acc[global_id] = functor(
                leaf_table_acc[leaf_idx * local_stored_leaf_size + local_id],
                q_point);
            //            dist_acc[global_id] = functor(
            //              leaf_table_acc[0],
            //              local_q_points_ref[0]);
          });
    });
  }

  // Write Back the results
  for (int i = 0; i < stored_num_batches; ++i) {
    std::for_each(dists.begin() + i * stored_leaf_size,
                  dists.begin() + i * stored_leaf_size + stored_leaf_size,
                  [&](auto dist) { result_sets[i].AddPoint(dist); });
  }
}

void ExecuteBatchedKernelsAsync(long tid) {
  // At this point, buffers are filled.
  // const auto num_leaf_to_process =
  // batches[cur_collecting].next_avalible_slot; std::cout <<
  // "num_leaf_to_process: " << stored_num_batches << std::endl;

  constexpr auto functor = MyFunctor();

  // std::vector<float> dists(total_num_items);

  ProcessBatchSycl(q[cur_collecting], batches[cur_collecting]);

  // {
  //   sycl::buffer dists_buf(dists);  // 1024 * 1024
  //   sycl::buffer buffer_buf(batch[cur_collecting].u_buffer,
  //   stored_num_batches);

  //   q[cur_collecting].submit([&](auto& h) {
  //     sycl::accessor leaf_table_acc(*b_node_data_buf, h, sycl::read_only);
  //     sycl::accessor query_table_acc(*b_query_point_buf, h, sycl::read_only);

  //     sycl::accessor buffer_acc(buffer_buf, h, sycl::read_only);

  //     sycl::accessor dist_acc(dists_buf, h, sycl::write_only, sycl::no_init);

  //     h.parallel_for(sycl::nd_range<1>(num_total_works, kBlockThreads),
  //                    [=](const sycl::nd_item<1> item) {
  //                      const auto global_id = item.get_global_id(0);
  //                      const auto group_id = item.get_group(0);  // 0...4095
  //                      / 4

  //                     //buffer_acc;

  //                     //  for (int i = 0; i < items_per_thread; ++i) {
  //                     //    dist_acc[global_id * items_per_thread + i] =
  //                     functor(
  //                     //        leaf_table_acc[global_id * items_per_thread +
  //                     i],
  //                     //        query_table_acc[group_id]);
  //                     //  }
  //                    });
  //   });
  // }

  // for (int i = 0; i < stored_num_batches; ++i) {
  //   for (int j = 0; j < kK; ++j) {
  //     result_sets[i].AddPoint(dists[i * kK + j]);
  //   }
  // }

  // Switch
  auto next_collecting = (cur_collecting + 1) % kNumStreams;
  q[next_collecting].wait();
  cur_collecting = next_collecting;
  batches[cur_collecting].Reset();
}

void EndReducer() {}

}  // namespace redwood