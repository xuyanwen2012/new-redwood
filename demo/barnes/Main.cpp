#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <vector>

#include "../../src/Utils.hpp"
#include "../PointCloud.hpp"
#include "../ThreadHelper.hpp"
#include "Executor.hpp"
#include "Kernel.cuh"
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

Point3F CpuNaiveQuery(const Point4F* in_data, const Point3F* q_data,
                      const unsigned n, const unsigned query_idx) {
  auto kernel_func = MyFunctor();
  Point3F sum{};
  const auto q = q_data[query_idx];
  for (auto i = 0u; i < n; ++i) {
    auto force = kernel_func(in_data[i], q);
    sum += force;
  }
  return sum;
}

int main() {
  const auto n = 4096 * 1024;  // better to be 8^x
  const auto m = 32 * 1024;
  const auto theta = 0.1f;
  const auto leaf_size = 256;
  const auto num_threads = 1;
  const auto num_batches = 256;  // For sycl
  const auto batch_size = 9440;  // From, 295*32

  std::cout << "Simulation Prameters:\n"
            << "\tn: " << n << '\n'
            << "\tm: " << m << '\n'
            << "\ttheta: " << theta << '\n'
            << "\tleaf_size: " << leaf_size << '\n'
            << "\tnum_threads: " << num_threads << '\n'
            << "\tnum_batches: " << num_batches << '\n'
            << "\tbatch_size: " << batch_size << '\n'
            << std::endl;

  assert(m % num_threads == 0);

  std::cout << "Preparing data... " << '\n';
  std::vector<Point4F> h_in_data(n);
  std::generate(h_in_data.begin(), h_in_data.end(),
                MakeLargeRandomPoint<4, float>);

  std::cout << "Building Tree... " << '\n';
  const oct::OctreeParams<float> params{
      theta,
      leaf_size,
      {Point3F{100.0f, 100.0f, 100.0f}, Point3F{50.0f, 50.0f, 50.0f}}};
  oct::Octree<float> tree(h_in_data.data(), n, params);
  tree.BuildTree();

  std::cout << "Preparing REDwood... " << '\n';
  redwood::InitReducer(num_threads, leaf_size, num_batches, batch_size);
  redwood::SetNodeTables(tree.GetLeafNodeTable().Data(),
                         tree.GetLeafSizeTable().Data(),
                         tree.GetStats().num_leaf_nodes);

  // Partition the data accross threads
  const auto num_task_per_thread = m / num_threads;
  std::vector<redwood::UnifiedContainer<Point3F>> u_q_data(num_threads);
  std::vector<redwood::dev::BarnesExecutorManager<float>> managers;
  managers.reserve(num_threads);

  for (int tid = 0; tid < num_threads; ++tid) {
    u_q_data[tid].Allocate(num_task_per_thread);
    std::generate(u_q_data[tid].begin(), u_q_data[tid].end(),
                  MakeLargeRandomPoint<3, float>);

    redwood::SetQueryPoints(tid, u_q_data[tid].Data(), num_task_per_thread);
    managers.emplace_back(tree, u_q_data[tid].Data(), num_task_per_thread,
                          num_batches, tid);
  }

  std::cout << "Started Traversal... " << '\n';
  TimeTask("Traversal", [&] {
    ParallelFor(managers.begin(), managers.end(),
                [&](auto& manager) { manager.StartTraversal(); });
    redwood::EndReducer();
  });

  for (int i = 0; i < num_threads; ++i) {
    const auto stats = managers[i].GetStats();
    std::cout << "\tBr: " << stats.branch_node_reduced
              << "\tLe: " << stats.leaf_node_reduced << std::endl;
  }

  const auto n_display = std::min(num_task_per_thread, 5);
  for (auto i = 0; i < n_display; ++i) {
    Point3F rst{};
    redwood::GetReductionResult(0l, i, &rst);

    std::cout << "Query " << i << ":\n"
              << "\tquery_point: \t" << u_q_data[0l][i] << '\n'
              << "\tforce:    \t" << rst << '\n';

    if constexpr (constexpr auto show_ground_truth = true) {
      std::cout << "\tground_truth: \t"
                << CpuNaiveQuery(h_in_data.data(), u_q_data[0l].Data(), n, i)
                << '\n';
    }
    std::cout << std::endl;
  }

  return EXIT_SUCCESS;
}