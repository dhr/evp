#pragma once
#ifndef EVP_UTIL_NDINDEX_H
#define EVP_UTIL_NDINDEX_H

#include <cassert>
#include <cstdarg>

#include <vector>

namespace evp {
using namespace clip;

template<i32 N>
class NDIndex {
 public:
  typedef i32 size_type;
  
 private:
  size_type elements_[N];
  
 public:
	explicit NDIndex() {
    for (size_type i = 0; i < N; ++i)
      elements_[i] = 0;
  }
  
	NDIndex(size_type e1, ...) {
    va_list ap;
    va_start(ap, e1);
		elements_[0] = e1;
    for (size_type i = 1; i < N; ++i)
      elements_[i] = va_arg(ap, i32);
    va_end(ap);
  }
  
  NDIndex(size_type* sizes) {
    for (size_type i = 0; i < N; ++i)
      elements_[i] = sizes[i];
  }
  
  inline size_type &operator[](size_type index) {
    return elements_[index];
  }
  
  inline const size_type &operator[](size_type index) const {
    return elements_[index];
  }
};

}

#endif
