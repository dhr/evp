#pragma once
#ifndef EVP_IO_NDARRAYIO_H
#define EVP_IO_NDARRAYIO_H

#include <clip.hpp>

#include "evp/util/ndarray.hpp"

namespace evp {
using namespace clip;

template<i32 N>
void ReadImageDataFromBufferArray(const NDArray<ImageBuffer,N> &buffers,
                                  NDArray<ImageData,N> &data) {
  i32 numElems = buffers.numElems();
  
  if (data.numElems() != numElems) {
    std::vector<i32> sizes(N);
    for (i32 i = 0; i < N; ++i)
      sizes[i] = buffers.size(i);
    
    data = NDArray<ImageData,N>(&sizes[0]);
  }
  
  for (i32 i = 0; i < numElems; ++i)
    data[i] = buffers[i].fetchData();
}

template<i32 N>
inline void WriteImageDataToBufferArray(const NDArray<ImageData,N> &data,
                                        NDArray<ImageBuffer,N> &buffers) {
  i32 numElems = data.numElems();
  
  if (buffers.numElems() != numElems) {
    std::vector<i32> sizes(N);
    for (i32 i = 0; i < N; ++i)
      sizes[i] = data.size(i);
    
    buffers = NDArray<ImageBuffer,N>(&sizes[0]);
  }
  
  for (i32 i = 0; i < numElems; ++i)
    buffers[i] = ImageBuffer(data[i]);
}

}

#endif
