#include <algorithm>
#include <array>
#include <iostream>
#include <vector>

#include "../../src/Redwood.hpp"
#include "../PointCloud.hpp"
#include "../ThreadHelper.hpp"
#include "../cxxopts.hpp"
#include "../nn/KDTree.hpp"
#include "../nn/Kernel.hpp"
#include "KnnExecutor.hpp"

std::vector<float> CpuNaiveQuery(const Point4F* in_data, const Point4F q,
                                 const unsigned n, const int k) {
  constexpr auto kernel_func = kernel::MyFunctor();

  std::vector<float> dists(n);
  std::transform(in_data, in_data + n, dists.begin(),
                 [&](const auto& p) { return kernel_func(p, q); });
  std::sort(dists.begin(), dists.end());
  return std::vector(dists.begin(), dists.begin() + k);
}

int main(int argc, char* argv[]) {
  cxxopts::Options options("KNN", "Heterogeneous Computing KNN Problem");
  options.add_options()("n,num", "Number of particles",
                        cxxopts::value<int>()->default_value("1048576"))(
      "p,thread", "Num Thread", cxxopts::value<int>()->default_value("1"))(
      "l,leaf", "Leaf node size",
      cxxopts::value<unsigned>()->default_value("1024"))(
      "q,query", "Num to Query", cxxopts::value<int>()->default_value("16384"))(
      "b,num_batch", "Num Batch", cxxopts::value<int>()->default_value("512"));

  const auto result = options.parse(argc, argv);
  const auto n = result["num"].as<int>();
  const auto m = result["query"].as<int>();
  constexpr auto k = 32;
  const auto leaf_size = result["leaf"].as<unsigned>();
  const auto num_threads = result["thread"].as<int>();
  const auto num_batches = result["num_batch"].as<int>();

  std::cout << "Simulation Prameters:\n"
            << "\tn: " << n << '\n'
            << "\tm: " << m << '\n'
            << "\tk: " << k << '\n'
            << "\tleaf_size: " << leaf_size << '\n'
            << "\tnum_threads: " << num_threads << '\n'
            << "\tnum_batches: " << num_batches << '\n'
            << std::endl;

  assert(m % num_threads == 0);

  std::cout << "Preparing dataset... " << '\n';
  std::vector<Point4F> h_in_data(n);
  std::generate(h_in_data.begin(), h_in_data.end(), MakeRandomPoint<4, float>);

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

    std::vector<std::vector<redwood::Task>> tasks_to_do(num_threads);
    const auto num_task_per_thread = m / num_threads;

    for (int tid = 0; tid < num_threads; ++tid) {
      // Generate sub tasks for each thread
      tasks_to_do[tid].resize(num_task_per_thread);
      for (int i = 0; i < num_task_per_thread; ++i) {
        tasks_to_do[tid][i].query_point = MakeRandomPoint<4, float>();
        tasks_to_do[tid][i].query_idx = i;
      }
      std::reverse(tasks_to_do[tid].begin(), tasks_to_do[tid].end());

      redwood::SetQueryPoints(tid, nullptr, num_task_per_thread);
    }

    using KnnManager =
        redwood::dev::ExecutorManager<float, redwood::dev::KnnExecutor>;
    std::vector<KnnManager> managers;
    managers.reserve(num_threads);
    for (int tid = 0; tid < num_threads; ++tid) {
      managers.emplace_back(kdt_ptr, tasks_to_do[tid], num_batches, tid);
    }

    std::cout << "Started Traversal... " << '\n';
    TimeTask("Traversal", [&] {
      ParallelFor(
          managers.begin(), managers.end(),
          [&](auto& manager) { manager.StartTraversals(); }, num_threads);
      redwood::EndReducer();
    });

    // Display results, and verify correctness
    const auto n_display = std::min(num_task_per_thread, 3);
    for (auto i = 0; i < n_display; ++i) {
      std::cout << "Query " << i << ":\n";

      if constexpr (kRedwoodBackend != redwood::Backends::kCpu) {
        float* result_set;
        redwood::GetReductionResult(0, i, &result_set);

        std::sort(result_set, result_set + k);

        for (int j = 0; j < k; ++j) {
          std::cout << "\t" << j << ":\t"
                    << result_set[j]  //<< " --- " << gt[j]
                    << '\n';
        }
      }
      std::cout << std::endl;
    }
#endif
  } else {
    // -------------- CPU ------------------------
    std::vector<std::vector<redwood::Task>> tasks_to_do(num_threads);
    const auto num_task_per_thread = m / num_threads;

    for (int tid = 0; tid < num_threads; ++tid) {
      // Generate sub tasks for each thread
      tasks_to_do[tid].resize(num_task_per_thread);
      for (int i = 0; i < num_task_per_thread; ++i) {
        tasks_to_do[tid][i].query_point = MakeRandomPoint<4, float>();
        tasks_to_do[tid][i].query_idx = i;
      }
      std::reverse(tasks_to_do[tid].begin(), tasks_to_do[tid].end());
    }

    using KnnManager = redwood::dev::SequentialManager<float>;
    std::vector<KnnManager> managers;
    managers.reserve(num_threads);
    for (int tid = 0; tid < num_threads; ++tid) {
      managers.emplace_back(kdt_ptr, tasks_to_do[tid], tid);
    }

    std::cout << "Started Cpu Traversal... " << '\n';
    TimeTask("Traversal", [&] {
      ParallelFor(
          managers.begin(), managers.end(),
          [&](auto& manager) { manager.StartTraversals(); }, num_threads);
    });

    const auto n_display = std::min(num_task_per_thread, 3);
    for (auto i = 0; i < n_display; ++i) {
      std::cout << "Query " << i << ":\n";

      const auto cpu_result = managers[0].GetCpuResult(i);

      for (int j = 0; j < k; ++j) {
        std::cout << "\t" << j << ":\t" << cpu_result[j] << '\n';
      }
      std::cout << std::endl;
    }
  }
  return EXIT_SUCCESS;
}
