#pragma once

#include <chrono>
#include <iostream>
#include <random>

namespace redwood {
enum class Backends {
  kCpu = 0,
  kCuda,
  kSycl,
  kDuet,
};
}

#ifdef REDWOOD_BACKEND
#if REDWOOD_BACKEND == 1
#define REDWOOD_IN_CUDA
constexpr auto kRedwoodBackend = redwood::Backends::kCuda;
#elif REDWOOD_BACKEND == 2
constexpr auto kRedwoodBackend = redwood::Backends::kSycl;
#elif REDWOOD_BACKEND == 3
constexpr auto kRedwoodBackend = redwood::Backends::kDuet;
#endif
#else
constexpr auto kRedwoodBackend = redwood::Backends::kCpu;
#define REDWOOD_IN_CPU
#endif

#ifdef REDWOOD_IN_CUDA
#define _REDWOOD_KERNEL __host__ __device__
#else
#define _REDWOOD_KERNEL
#endif

#define NO_OP 0

#ifndef _HAS_NODISCARD
#ifndef __has_cpp_attribute
#define _HAS_NODISCARD 0
#elif __has_cpp_attribute(nodiscard) >= \
    201603L  // TRANSITION, VSO#939899 (need toolset update)
#define _HAS_NODISCARD 1
#else
#define _HAS_NODISCARD 0
#endif
#endif  // _HAS_NODISCARD

#if _HAS_NODISCARD
#define _NODISCARD [[nodiscard]]
#else  // ^^^ CAN HAZ [[nodiscard]] / NO CAN HAZ [[nodiscard]] vvv
#define _NODISCARD
#endif  // _HAS_NODISCARD

template <typename T>
T my_rand(T min = 0.0, T max = 1.0) {
  // 114514 and 233
  static thread_local std::mt19937 generator(114514);  // NOLINT(cert-msc51-cpp)
  std::uniform_real_distribution<T> distribution(min, max);
  return distribution(generator);
}

template <typename T>
T MyRoundUp(T num_to_round, T multiple = 32) {
  T remainder = num_to_round % multiple;
  if (remainder == 0) return num_to_round;

  return num_to_round + multiple - remainder;
}

template <typename Func>
void TimeTask(const std::string& task_name, Func&& f) {
  const auto t0 = std::chrono::high_resolution_clock::now();

  std::forward<Func>(f)();

  const auto t1 = std::chrono::high_resolution_clock::now();
  const auto time_span =
      std::chrono::duration_cast<std::chrono::duration<float>>(t1 - t0);

  std::cout << "Finished " << task_name << "! Time took: " << time_span.count()
            << "s. " << std::endl;
}