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

int main() {
  const auto n = 1 << 20;
  const auto m = 64 * 1024;

  const auto leaf_size = 512;

  const auto num_threads = 1;
  const auto num_batches = 1024;  // each batch takes a leaf, and a single query

  std::cout << "Simulation Prameters:\n"
            << "\tn: " << n << '\n'
            << "\tm: " << m << '\n'
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

  std::cout << "Preparing REDwood... " << '\n';
  redwood::InitReducer(num_threads, leaf_size, num_batches);
  redwood::SetNodeTables(kdt_ptr->GetNodeContentTable().data(),
                         kdt_ptr->GetStats().num_leaf_nodes);

  // Partition the data accross threads
  const auto num_task_per_thread = m / num_threads;
  std::vector<redwood::UnifiedContainer<Point4F>> u_q_data(num_threads);
  std::vector<redwood::dev::NnCpuManager<float>> managers;
  managers.reserve(num_threads);

  for (int tid = 0; tid < num_threads; ++tid) {
    // Generate sub Query Points for each CPU thread
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

  const auto n_display = std::min(num_task_per_thread, 5);
  for (auto i = 0; i < n_display; ++i) {
    std::cout << "Query " << i << ":\n"
              << "\tquery_point: \t" << u_q_data[0l][i] << '\n'
              << "\tresult:      \t" << managers[0].GetCpuResult(i) << '\n';
    std::cout << std::endl;
  }

  return EXIT_SUCCESS;
}