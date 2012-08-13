#pragma once
#ifndef EVP_IO_NDARRAYIO_H
#define EVP_IO_NDARRAYIO_H

#include <clip.hpp>

#include "evp/util/memutil.hpp"
#include "evp/util/ndarray.hpp"

namespace evp {
using namespace clip;

template<i32 N>
std::tr1::shared_ptr< NDArray<ImageData,N> >
BufferArrayToDataArray(const NDArray<ImageBuffer,N>& buffers) {
  std::tr1::shared_ptr< NDArray<ImageData,N> > dataPtr
    (new NDArray<ImageData,N>(buffers.sizes()));
  NDArray<ImageData,N>& data = *dataPtr;
  
  i32 numElems = buffers.numElems();
  for (i32 i = 0; i < numElems; ++i)
    data[i] = buffers[i].fetchData();
  
  return dataPtr;
}

template<i32 N>
std::tr1::shared_ptr< NDArray<ImageBuffer,N> >
DataArrayToBufferArray(const NDArray<ImageData,N>& data) {
  i32 numElems = data.numElems();
  std::tr1::shared_ptr< NDArray<ImageBuffer,N> > outputPtr
    (new NDArray<ImageBuffer,N>(data.sizes()));
  NDArray<ImageBuffer,N>& buffers = *outputPtr;
  
  for (i32 i = 0; i < numElems; ++i)
    buffers[i] = ImageBuffer(data[i]);
  return outputPtr;
}

}

#endif
