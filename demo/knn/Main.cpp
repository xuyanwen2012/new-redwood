#include <algorithm>
#include <array>
#include <iostream>

#include "../../src/Redwood.hpp"
#include "../PointCloud.hpp"
#include "../nn/KDTree.hpp"
#include "../nn/Kernel.hpp"
#include "KnnExecutor.hpp"

std::vector<float> CpuNaiveQuery(const Point4F* in_data, const Point4F q,
                                 const unsigned n, const int k) {
  constexpr auto kernel_func = MyFunctor();

  std::vector<float> dists(n);
  std::transform(in_data, in_data + n, dists.begin(),
                 [&](const auto& p) { return kernel_func(p, q); });
  std::sort(dists.begin(), dists.end());
  return std::vector(dists.begin(), dists.begin() + k);
}

int main() {
  const auto n = 1 << 20;
  const auto m = 16 * 1024;
  constexpr auto k = 32;
  const auto leaf_size = 128;
  const auto num_batches = 1024;  // For SYCL, 512 is better
  const auto num_threads = 1;
  std::cout << "Simulation Prameters:\n"
            << "\tn: " << n << '\n'
            << "\tm: " << m << '\n'
            << "\tk: " << k << '\n'
            << "\tleaf_size: " << leaf_size << '\n'
            << "\tnum_threads: " << num_threads << '\n'
            << "\tnum_batches: " << num_batches << '\n'
            << std::endl;

  std::cout << "Preparing dataset... " << '\n';
  std::vector<Point4F> h_in_data(n);
  std::generate(h_in_data.begin(), h_in_data.end(), MakeRandomPoint<4, float>);

  const auto tid = 0;
  const auto num_task_per_thread = m / 1;

  std::vector<redwood::Task> tasks_to_do(num_task_per_thread);
  for (int i = 0; i < num_task_per_thread; ++i) {
    tasks_to_do[i].query_point = MakeRandomPoint<4, float>();
    tasks_to_do[i].query_idx = i;
  }
  const std::vector for_display(tasks_to_do.begin(), tasks_to_do.begin() + 5);
  std::reverse(tasks_to_do.begin(), tasks_to_do.end());

  std::cout << "Building KD Tree... " << '\n';
  const kdt::KdtParams params{leaf_size};
  auto kdt_ptr = std::make_shared<kdt::KdTree>(params, h_in_data);
  kdt_ptr->BuildTree();

  const auto h_lnd = kdt_ptr->GetNodeContentTable().data();
  const auto num_leaf_nodes = kdt_ptr->GetStats().num_leaf_nodes;

  // Actually use redwood here
  if constexpr (kRedwoodBackend != redwood::Backends::kCpu) {
#ifndef REDWOOD_IN_CPU
    std::cout << "Preparing REDwood... " << '\n';
    redwood::InitReducer(num_threads, leaf_size, num_batches);
    redwood::SetNodeTables(h_lnd, num_leaf_nodes);
    redwood::SetQueryPoints(tid, nullptr, num_task_per_thread);

    using KnnManager =
        redwood::dev::ExecutorManager<float, redwood::dev::KnnExecutor>;
    std::vector<KnnManager> managers;
    managers.reserve(num_threads);
    managers.emplace_back(kdt_ptr, tasks_to_do, num_batches, tid);

    std::cout << "Started Traversal... " << '\n';
    TimeTask("Traversal", [&] {
      managers[tid].StartTraversals();
      redwood::EndReducer();
    });
#endif
  } else {
    // Cpu
    std::vector<redwood::dev::SequentialManager<float>> managers;
    managers.reserve(num_threads);
    managers.emplace_back(kdt_ptr, tasks_to_do, tid);

    std::cout << "Started Cpu Traversal... " << '\n';
    TimeTask("Traversal", [&] { managers[tid].StartTraversals(); });

    const auto n_display = std::min(num_task_per_thread, 3);
    for (auto i = 0; i < n_display; ++i) {
      std::cout << "Query " << i << ":\n"
                << "\tQuery point " << for_display[i].query_point << '\n';

      auto result_set = managers[tid].GetCpuResult(i);
      for (int j = 0; j < k; ++j) {
        std::cout << "\t" << j << ":\t" << result_set[j] << '\n';
      }
    }

    return EXIT_SUCCESS;
  }

  // Display results, and verify correctness
  const auto n_display = std::min(num_task_per_thread, 3);
  for (auto i = 0; i < n_display; ++i) {
    std::cout << "Query " << i << ":\n"
              << "\tQuery point " << for_display[i].query_point << '\n';

    if constexpr (kRedwoodBackend != redwood::Backends::kCpu) {
      float* result_set;
      redwood::GetReductionResult(0, i, &result_set);

      std::sort(result_set, result_set + k);

      auto gt =
          CpuNaiveQuery(h_in_data.data(), for_display[i].query_point, n, k);
      for (int j = 0; j < k; ++j) {
        std::cout << "\t" << j << ":\t" << result_set[j] << " --- " << gt[j]
                  << '\n';
      }
    }
    std::cout << std::endl;
  }

  return EXIT_SUCCESS;
}
