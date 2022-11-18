#include <CL/sycl.hpp>
#include <cstdlib>
#include <vector>

#include "../PointCloud.hpp"

sycl::device device;
sycl::context ctx;
sycl::property_list props;
int stored_num_batches;
int stored_batch_size;

void InitSycl() {
  // Intel(R) UHD Graphics [0x9b41] on Parakeet
  device = sycl::device::get_devices(sycl::info::device_type::all)[1];
  std::cout << "SYCL Device: " << device.get_info<sycl::info::device::name>()
            << std::endl;

  props = {sycl::property::buffer::use_host_ptr()};
}

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
    stored_num_batches = num;
    stored_batch_size = size;

    const auto bytes = sizeof(Point4F) * num * size;
    u_data =
        static_cast<Point4F*>(sycl::malloc_shared(bytes, device, ctx, props));

    const auto unsigned_bytes = sizeof(unsigned) * num;
    u_query_idx = static_cast<unsigned*>(
        sycl::malloc_shared(unsigned_bytes, device, ctx, props));

    u_items_in_batch = static_cast<unsigned*>(
        sycl::malloc_shared(unsigned_bytes, device, ctx, props));
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
    // Batch overflow, use the next one
    if (current_idx_in_batch == stored_batch_size) {
      u_items_in_batch[current_batch] = current_idx_in_batch;
      ++current_batch;

      u_query_idx[current_batch] = q_idx;
      current_idx_in_batch = 0;
    }

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

int main() {
  InitSycl();
  std::vector<Point4F> test(1024);

  for (int i = 0; i < 1024; ++i) {
    test[i].data[0] = i;
  }

  const auto num = 4;
  const auto size = 16;
  BarnesBranchBatch small_batch;
  small_batch.AllocateBuffer(num, size);

  auto jjjj = 0;

  for (int j = 0; j < 2; ++j) {
    small_batch.OnStartQuery(j);
    for (int i = 0; i < 28; ++i) {
      small_batch.LoadBranchNode(j, &test[jjjj++]);
    }
  }
  small_batch.EndTraversal();

  for (int i = 0; i < num; ++i) {
    std::cout << i << ": " << small_batch.u_query_idx[i] << "\t"
              << small_batch.u_items_in_batch[i] << std::endl;
    for (int j = 0; j < size; ++j) {
      std::cout << "\t" << j << ": " << small_batch.u_data[i * size + j]
                << std::endl;
    }
  }

  return EXIT_SUCCESS;
}