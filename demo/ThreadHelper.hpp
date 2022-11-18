#pragma once

#include <algorithm>
#include <thread>
#include <vector>

// TODO: should also be able to use OMP

template <typename Iterator, class Function>
void ParallelFor(const Iterator& first, const Iterator& last, Function&& f,
                 const int nthreads = 1) {
  // Each 'ExecutorManager' will be in each thread.
  constexpr unsigned group = 1;
  std::vector<std::thread> threads;

  threads.reserve(nthreads);
  Iterator it = first;
  for (; it < last - group; it += group) {
    threads.push_back(std::thread(
        [=, &f]() { std::for_each(it, std::min(it + group, last), f); }));
  }
  std::for_each(it, last, f);  // last steps while we wait for other threads

  std::for_each(threads.begin(), threads.end(),
                [](std::thread& x) { x.join(); });
}
