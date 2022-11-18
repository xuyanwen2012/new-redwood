#pragma once

#include <cstdlib>
#include <iterator>
#include <stdexcept>

#include "Utils.hpp"

namespace redwood {

namespace internal {
void AllocateUnified(void *data, int bytes);
void DeallocateUnified(void *data);
}  // namespace internal

template <typename DataT>
class UnifiedContainer {
 public:
  UnifiedContainer() = default;
  UnifiedContainer(const int n) : size_(n) { Allocate(n); };

  void Allocate(const int n) {
    size_ = n;
    u_data_ = static_cast<DataT *>(aligned_alloc(64, sizeof(DataT) * n));
  }

  ~UnifiedContainer() { free(u_data_); }

  DataT &operator[](int i) {
    if constexpr (kDebugMode)
      if (i >= size_)
        throw std::runtime_error("UnifiedContainer::operator[] Out of Bounds");

    return u_data_[i];
  }
  const DataT &operator[](int i) const {
    if constexpr (kDebugMode)
      if (i >= size_)
        throw std::runtime_error("UnifiedContainer::operator[] Out of Bounds");
    return u_data_[i];
  }

  DataT *Data() noexcept { return u_data_; };
  const DataT *Data() const noexcept { return u_data_; };
  std::size_t Size() const { return size_; }

 private:
  DataT *u_data_;
  int size_;

 public:
  struct Iterator {
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = DataT;
    using pointer = DataT *;
    using reference = DataT &;

    Iterator(pointer ptr) : m_ptr(ptr) {}
    reference operator*() const { return *m_ptr; }
    pointer operator->() { return m_ptr; }

    Iterator &operator++() {
      m_ptr++;
      return *this;
    }

    friend bool operator==(const Iterator &a, const Iterator &b) {
      return a.m_ptr == b.m_ptr;
    };
    friend bool operator!=(const Iterator &a, const Iterator &b) {
      return a.m_ptr != b.m_ptr;
    };

   private:
    pointer m_ptr;
  };

  Iterator begin() { return Iterator(u_data_); }
  Iterator end() { return Iterator(u_data_ + size_); }
};

}  // namespace redwood