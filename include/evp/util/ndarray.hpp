#pragma once
#ifndef EVP_UTIL_NDARRAY_H
#define EVP_UTIL_NDARRAY_H

#include <cassert>
#include <cstdarg>

#include <algorithm>
#include <vector>

#include "evp/util/ndindex.hpp"

namespace evp {
using namespace clip;

template<typename T, i32 N>
class NDArray {
 public:
  typedef size_t size_type;
  typedef typename std::vector<T>::iterator iterator;
  
 private:
  size_type sizes_[N];
  std::vector<T> elements_;
  
 public:
  NDArray() : elements_(1) {
    std::fill_n(sizes_, N, 1);
  }
 
  explicit NDArray(T init) : elements_(1, init) {
    std::fill_n(sizes_, N, 1);
  }
  
  NDArray(const size_type *const sizes) {
    size_type numElems = 1;
    for (size_type i = 0; i < N; i++) {
      sizes_[i] = sizes[i];
      numElems *= sizes_[i];
    }
    
    elements_.resize(numElems);
  }
  
  NDArray(size_type size1, ...) {
    va_list ap;
    va_start(ap, size1);
    sizes_[0] = size1;
    size_type numElems = size1;
    for (size_type i = 1; i < N; ++i) {
      sizes_[i] = va_arg(ap, size_type);
      numElems *= sizes_[i];
    }
    va_end(ap);
    
    elements_.resize(numElems);
  }
  
  inline size_type numElems() const { return size_type(elements_.size()); }
  
  inline i32 size(size_type dim) const {
    assert(dim >= 0 && dim < N && "Dimension index out of bounds");
    return sizes_[dim];
  }
  
  inline const size_type* sizes() const {
    return sizes_;
  }
  
  inline T &operator()(size_type index, ...) {
    va_list ap;
    va_start(ap, index);
    i32 skip = sizes_[0];
    for (size_type i = 1; i < N; ++i) {
      index += va_arg(ap, i32)*skip;
      skip *= sizes_[i];
    }
    va_end(ap);
    
    return elements_[index];
  }
  
  const inline T &operator()(size_type index, ...) const {
    va_list ap;
    va_start(ap, index);
    size_type skip = sizes_[0];
    for (size_type i = 1; i < N; ++i) {
      index += va_arg(ap, i32)*skip;
      skip *= sizes_[i];
    }
    va_end(ap);
    
    return elements_[index];
  }
  
  inline size_type equivalentIndex(const NDIndex<N> &indices) const {
    size_type index = indices[0];
    size_type skip = sizes_[0];
    for (size_type i = 1; i < N; i++) {
      index += indices[i]*skip;
      skip *= sizes_[i];
    }
    
    assert(index < numElems());
    
    return index;
  }
  
  inline T &operator[](const size_type index) {
    return elements_[index];
  }
  
  inline const T &operator[](const size_type index) const {
    return elements_[index];
  }
  
  T &operator[](const NDIndex<N> &indices) {
    i32 index = equivalentIndex(indices);
    return elements_[index];
  }
  
  const T &operator[](const NDIndex<N> &indices) const {
    i32 index = equivalentIndex(indices);
    return elements_[index];
  }
  
  iterator begin() {
    return elements_.begin();
  }
  
  iterator end() {
    return elements_.end();
  }
};

}

#endif
