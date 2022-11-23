#include <algorithm>
#include <array>
#include <cassert>
#include <vector>

#include "../../src/Containers.hpp"
#include "../../src/Utils.hpp"
#include "../PointCloud.hpp"
#include "../ThreadHelper.hpp"
#include "Executor.hpp"
#include "KDTree.hpp"
#include "Kernel.cuh"

template <int K>
std::array<float, K> CpuNaiveQuery(const Point4F* in_data, const Point4F q,
                                   const unsigned n) {
  constexpr auto kernel_func = MyFunctor();

  std::vector<float> dists(n);
  std::transform(in_data, in_data + n, dists.begin(),
                 [&](const auto& p) { return kernel_func(p, q); });

  std::sort(dists.begin(), dists.end());

  std::array<float, K> result;
  std::copy_n(dists.begin(), K, result.begin());
  return result;
}

int main() {
  const auto n = 1 << 20;
  const auto m = 16 * 1024;

  const auto leaf_size = 32;
  const auto k = 32;

  const auto num_threads = 1;
  const auto num_batches = 1024;  // each batch takes a leaf, and a single query

  std::cout << "Simulation Prameters:\n"
            << "\tn: " << n << '\n'
            << "\tm: " << m << '\n'
            << "\tk: " << k << '\n'
            << "\tleaf_size: " << leaf_size << '\n'
            << "\tnum_threads: " << num_threads << '\n'
            << "\tnum_batches: " << num_batches << '\n'
            << std::endl;

  assert(m % num_threads == 0);

  std::cout << "Preparing data... " << '\n';
  std::vector<Point4F> h_in_data(n);
  std::generate(h_in_data.begin(), h_in_data.end(), MakeRandomPoint<4, float>);

  // Build KD Tree
  std::cout << "Building KD Tree... " << '\n';
  const kdt::KdtParams params{leaf_size};
  auto kdt_ptr = std::make_shared<kdt::KdTree>(params, h_in_data);
  kdt_ptr->BuildTree();

  // Partition the data accross threads
  const auto num_task_per_thread = m / num_threads;
  std::vector<redwood::UnifiedContainer<Point4F>> u_q_data(num_threads);
  std::vector<redwood::dev::KnnCpuManager<float>> managers;
  managers.reserve(num_threads);

  for (int tid = 0; tid < num_threads; ++tid) {
    u_q_data[tid].Allocate(num_task_per_thread);
    std::generate(u_q_data[tid].begin(), u_q_data[tid].end(),
                  MakeRandomPoint<4, float>);

    redwood::SetQueryPoints(tid, u_q_data[tid].Data(), num_task_per_thread);
    managers.emplace_back(kdt_ptr, u_q_data[tid].Data(), num_task_per_thread,
                          tid);
  }

  std::cout << "Started Traversal... " << '\n';
  TimeTask("Traversal", [&] {
    ParallelFor(managers.begin(), managers.end(),
                [&](auto& manager) { manager.StartTraversals(); });
    redwood::EndReducer();
  });

  for (int i = 0; i < num_threads; ++i) {
    const auto stats = managers[i].GetStats();
    std::cout << "\tBr: " << stats.branch_node_reduced
              << "\tLe: " << stats.leaf_node_reduced << std::endl;
  }

  const auto n_display = std::min(num_task_per_thread, 2);
  for (auto i = 0; i < n_display; ++i) {
    const auto rst = managers[0].GetCpuResult(i, k);

    std::cout << "Query " << i << ":\n"
              << "\tquery_point: \t" << u_q_data[0l][i] << '\n'
              << "\tResults:\n";

    if constexpr (constexpr auto show_ground_truth = true) {
      auto gt = CpuNaiveQuery<32>(h_in_data.data(), u_q_data[0l][i], n);

      for (int i = 0; i < k; ++i) {
        std::cout << "\t\t" << rst[i] << "\t---\t" << gt[i] << "\n";
      }

    } else {
      for (int i = 0; i < k; ++i) {
        std::cout << "\t\t" << rst[i] << "\n";
      }
    }
    std::cout << std::endl;
  }

  return EXIT_SUCCESS;
}