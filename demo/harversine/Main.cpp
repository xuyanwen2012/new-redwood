#include <algorithm>
#include <array>
#include <iostream>

#include "../../src/Redwood.hpp"
#include "../PointCloud.hpp"
#include "../cxxopts.hpp"
#include "KDTree.hpp"
#include "Kernel.hpp"
#include "NnExecutor.hpp"

float CpuNaiveQuery(const Point2F* in_data, const Point2F q, const unsigned n) {
  constexpr auto kernel_func = MyFunctor();

  std::vector<float> dists(n);
  std::transform(in_data, in_data + n, dists.begin(),
                 [&](const auto& p) { return kernel_func(p, q); });

  return *std::min_element(dists.begin(), dists.end());
}

int main(int argc, char* argv[]) {
  cxxopts::Options options("Barnes Hut",
                           "Heterogeneous Computing N-Body Problem");
  options.add_options()("n,num", "Number of particles",
                        cxxopts::value<int>()->default_value("1048576"))(
      "p,thread", "Num Thread", cxxopts::value<int>()->default_value("1"))(
      "l,leaf", "Leaf node size",
      cxxopts::value<unsigned>()->default_value("1024"))(
      "q,query", "Num to Query", cxxopts::value<int>()->default_value("65536"))(
      "b,num_batch", "Num Batch", cxxopts::value<int>()->default_value("512"));

  const auto result = options.parse(argc, argv);
  const auto n = result["num"].as<int>();
  const auto m = result["query"].as<int>();
  const auto leaf_size = result["leaf"].as<unsigned>();
  const auto num_threads = result["thread"].as<int>();
  const auto num_batches = result["num_batch"].as<int>();

  std::cout << "Simulation Prameters:\n"
            << "\tn: " << n << '\n'
            << "\tm: " << m << '\n'
            << "\tleaf_size: " << leaf_size << '\n'
            << "\tnum_threads: " << num_threads << '\n'
            << "\tnum_batches: " << num_batches << '\n'
            << std::endl;

  // assert(m % num_threads == 0);

  static const auto make_random_location = [] {
    return Point2F{my_rand<float>(0.0f, 90.0f), my_rand<float>(0.0f, 180.0f)};
  };

  std::cout << "Preparing dataset... " << '\n';
  std::vector<Point2F> h_in_data(n);
  std::generate(h_in_data.begin(), h_in_data.end(), make_random_location);

  const auto tid = 0;
  const auto num_task_per_thread = m / 1;

  std::vector<redwood::Task> tasks_to_do(num_task_per_thread);
  for (int i = 0; i < num_task_per_thread; ++i) {
    tasks_to_do[i].query_point = make_random_location();
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

    using NnManager =
        redwood::dev::ExecutorManager<float, redwood::dev::NnExecutor>;
    std::vector<NnManager> managers;
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
      auto result = managers[tid].GetCpuResult(i);

      std::cout << "Query " << i << ":\n"
                << "\tQuery point " << for_display[i].query_point << '\n'
                << "\tResult " << result << '\n';

      if constexpr (constexpr auto show_ground_truth = true) {
        std::cout << "\tground_truth: \t"
                  << CpuNaiveQuery(h_in_data.data(), for_display[i].query_point,
                                   n)
                  << '\n';
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
      float* rst;
      redwood::GetReductionResult(0l, i, &rst);

      std::cout << "Query " << i << ":\n"
                << "\tquery_point: \t" << for_display[i].query_point << '\n'
                << "\tresult:      \t" << *rst << '\n';

      if constexpr (constexpr auto show_ground_truth = true) {
        std::cout << "\tground_truth: \t"
                  << CpuNaiveQuery(h_in_data.data(), for_display[i].query_point,
                                   n)
                  << '\n';
      }
      std::cout << std::endl;
    }
  }
  return EXIT_SUCCESS;
}