#include <algorithm>
#include <array>
#include <iostream>

#include "../../src/Redwood.hpp"
#include "../PointCloud.hpp"
#include "../cxxopts.hpp"
#include "Executor.hpp"
#include "Kernel.hpp"
#include "Octree.hpp"

template <int Dim, typename T>
Point<Dim, T> MakeLargeRandomPoint() {
  Point<Dim, T> pt;
  for (int i = 0; i < Dim - 1; ++i) {
    pt.data[i] = my_rand(T(0.0), T(100.0));
  }
  // Mass should be much smaller
  pt.data[Dim - 1] = T(1.0) + my_rand<T>();

  return pt;
}

template <typename T>
Point<3, T> MakeLargeQueryPoint() {
  Point<3, T> pt;
  for (int i = 0; i < 3; ++i) {
    pt.data[i] = my_rand(T(0.0), T(100.0));
  }
  return pt;
}

Point3F CpuNaiveQuery(const Point4F* in_data, const Point3F q,
                      const unsigned n) {
  constexpr auto kernel_func = MyFunctor();
  Point3F sum{};
  for (auto i = 0u; i < n; ++i) {
    const auto force = kernel_func(in_data[i], q);
    sum += force;
  }
  return sum;
}

int main(int argc, char* argv[]) {
  cxxopts::Options options("Barnes Hut",
                           "Heterogeneous Computing N-Body Problem");
  options.add_options()("n,num", "Number of particles",
                        cxxopts::value<int>()->default_value("1048576"))(
      "t,theta", "Theta Value", cxxopts::value<float>()->default_value("0.1"))(
      "p,thread", "Num Thread", cxxopts::value<int>()->default_value("1"))(
      "l,leaf", "Leaf node size", cxxopts::value<unsigned>()->default_value("32"))(
      "q,query", "Num to Query", cxxopts::value<int>()->default_value("16384"))(
      "b,num_batch", "Num Batch", cxxopts::value<int>()->default_value("1024"))(
      "s,batch_size", "Batch Size",
      cxxopts::value<int>()->default_value("7936"));

  const auto result = options.parse(argc, argv);
  const auto n = result["num"].as<int>();
  const auto m = result["query"].as<int>();
  const auto leaf_size = result["leaf"].as<unsigned>();
  const auto theta = result["theta"].as<float>();
  const auto num_threads = result["thread"].as<int>();
  const auto num_batches = result["num_batch"].as<int>();
  const auto batch_size = result["batch_size"].as<int>();

  std::cout << "Simulation Prameters:\n"
            << "\tn: " << n << '\n'
            << "\tm: " << m << '\n'
            << "\ttheta: " << theta << '\n'
            << "\tleaf_size: " << leaf_size << '\n'
            << "\tnum_threads: " << num_threads << '\n'
            << "\tnum_batches: " << num_batches << '\n'
            << "\tbatch_size: " << batch_size << '\n'
            << std::endl;

  std::cout << "Preparing data... " << '\n';
  std::vector<Point4F> h_in_data(n);
  std::generate(h_in_data.begin(), h_in_data.end(),
                MakeLargeRandomPoint<4, float>);

  const auto tid = 0;
  const auto num_task_per_thread = m / 1;

  std::vector<redwood::Task<Point3F>> tasks_to_do(num_task_per_thread);
  for (int i = 0; i < num_task_per_thread; ++i) {
    tasks_to_do[i].query_point = MakeLargeQueryPoint<float>();
    tasks_to_do[i].query_idx = i;
  }
  const std::vector for_display(tasks_to_do.begin(), tasks_to_do.begin() + 5);
  std::reverse(tasks_to_do.begin(), tasks_to_do.end());

  std::cout << "Building Tree... " << '\n';
  const oct::OctreeParams params{
      theta,
      leaf_size,
      {Point3F{100.0f, 100.0f, 100.0f}, Point3F{50.0f, 50.0f, 50.0f}}};
  const auto tree_ptr =
      std::make_shared<oct::Octree<float>>(h_in_data.data(), n, params);
  tree_ptr->BuildTree();

  const auto h_lnd = tree_ptr->GetLeafNodeTable();
  const auto h_lns = tree_ptr->GetLeafSizeTable();
  const auto num_leaf_nodes = tree_ptr->GetStats().num_leaf_nodes;

  // Actually use redwood here
  std::cout << "Preparing REDwood... " << '\n';
  redwood::InitReducer(num_threads, leaf_size, num_batches);
  redwood::SetNodeTables(h_lnd, h_lns, num_leaf_nodes);

  std::vector<redwood::dev::BarnesExecutorManager> managers;
  managers.reserve(num_threads);
  managers.emplace_back(tree_ptr, tasks_to_do, num_batches, batch_size, tid);

  std::cout << "Started Traversal... " << '\n';
  TimeTask("Traversal", [&] {
    managers[0].StartTraversals();
    redwood::EndReducer();
  });

  // Display results, and verify correctness
  const auto n_display = std::min(num_task_per_thread, 5);
  for (auto i = 0; i < n_display; ++i) {
    Point3F rst{};
    redwood::GetReductionResult(0l, i, &rst);

    std::cout << "Query " << i << ":\n"
              << "\tquery_point: \t" << for_display[i].query_point << '\n'
              << "\tforce:    \t" << rst << '\n';

    if constexpr (constexpr auto show_ground_truth = true) {
      std::cout << "\tground_truth: \t"
                << CpuNaiveQuery(h_in_data.data(), for_display[i].query_point,
                                 n)
                << '\n';
    }
    std::cout << std::endl;
  }

  return EXIT_SUCCESS;
}
