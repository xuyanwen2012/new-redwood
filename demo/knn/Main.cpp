#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <vector>

#include "../../src/Utils.hpp"
#include "../PointCloud.hpp"
#include "../ThreadHelper.hpp"
#include "KDTree.hpp"
#include "Kernel.cuh"

// Point3F CpuNaiveQuery(const Point4F* in_data, const Point3F* q_data,
//                       const unsigned n, const unsigned query_idx) {
//   auto kernel_func = MyFunctor();
//   Point3F sum{};
//   const auto q = q_data[query_idx];
//   for (auto i = 0u; i < n; ++i) {
//     auto force = kernel_func(in_data[i], q);
//     sum += force;
//   }
//   return sum;
// }

int main() {
  const auto n = 1 << 20;
  const auto m = 16 * 1024;

  const auto leaf_size = 256;
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



  return EXIT_SUCCESS;
}