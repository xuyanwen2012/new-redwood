

#include "../../src/Redwood.hpp"
#include "../PointCloud.hpp"
#include "../cxxopts.hpp"
#include "../harversine/KDTree.hpp"
#include "../harversine/Kernel.hpp"
// #include "NnExecutor.hpp"


int main(int argc, char* argv[]) {
    cxxopts::Options options("Haversine",
                           "Heterogeneous Computing Haversine NN Problem");
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
    return Point2D{my_rand<double>(0.0, 90.0), my_rand<double>(0.0, 180.0)};
  };

  std::cout << "Preparing dataset... " << '\n';
  std::vector<Point2D> h_in_data(n);
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


  


  return EXIT_SUCCESS;
}