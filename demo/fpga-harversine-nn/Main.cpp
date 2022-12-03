#include <fcntl.h>
#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

// #include "../../src/Redwood.hpp"
#include "../PointCloud.hpp"
#include "../harversine/Kernel.hpp"
// #include "../cxxopts.hpp"

constexpr auto kArg = 0;
constexpr auto kPos0X = 1;
constexpr auto kPos0Y = 2;
constexpr auto kPos0Z = 3;
constexpr auto kPos0W = 4;
constexpr auto kFincnt = 5;
constexpr auto kResult = 6;
constexpr auto kNEngine = 1;
volatile uint64_t* duet_baseaddr = nullptr;

int main(int argc, char* argv[]) {
  int fd = open("/dev/duet", O_RDWR);
  duet_baseaddr = static_cast<volatile uint64_t*>(mmap(
      nullptr, kNEngine << 13, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0));

  constexpr auto tid = 0;
  constexpr auto debug = true;

  // auto q = Point2D{12.0, 123.0};

  Point2D q;
  q.data[0] = 12.0;
  q.data[1] = 123.0;
  auto ptr = &q;

  std::vector<Point2D> leaf_node(64);
  for (int i = 0; i < 64; ++i) {
    leaf_node[i].data[0] = my_rand(0.0, 90.0);
    leaf_node[i].data[1] = my_rand(0.0, 180.0);
  }

  const long caller_id = tid;
  volatile uint64_t* sri = duet_baseaddr + (caller_id << 4) + 16;

  if constexpr (debug) {
    std::cout << tid << ": started duet. " << ptr->data[0] << std::endl;
  }

  sri[kPos0X] = *reinterpret_cast<const uint64_t*>(&ptr->data[0]);
  sri[kPos0Y] = *reinterpret_cast<const uint64_t*>(&ptr->data[1]);
  sri[kPos0Z] = *reinterpret_cast<const uint64_t*>(&ptr->data[2]);
  sri[kPos0W] = *reinterpret_cast<const uint64_t*>(&ptr->data[3]);

  auto node_base_addr = leaf_node.data();

  if constexpr (debug) {
    auto ptr = reinterpret_cast<const Point2D*>(node_base_addr);
    std::cout << tid << ": pushed duet. " << ptr->data[0]
              << "\taddress: " << node_base_addr << std::endl;
  }

  sri[kArg] = reinterpret_cast<uint64_t>(node_base_addr);

  if constexpr (debug) {
    std::cout << tid << ": poping. " << std::endl;
  }

  auto rzt = 0.0;
  auto result = &rzt;

  while (sri[kFincnt] < 1) NO_OP;

  auto addr = static_cast<double*>(result);
  auto tmp = *reinterpret_cast<const volatile double*>(&sri[kResult]);

  *addr = tmp;

  std::cout << rzt << std::endl;

  auto cpu_min = std::numeric_limits<double>::max();
  for (int i = 0; i < 64; ++i) {
    constexpr auto functor = MyFunctor();

    auto dist = functor(node_base_addr[i], q);
    cpu_min = std::min(cpu_min, dist);
  }

  std::cout << cpu_min << std::endl;

  return EXIT_SUCCESS;
}