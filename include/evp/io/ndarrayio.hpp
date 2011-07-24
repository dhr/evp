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
ReadImageDataFromBufferArray(const NDArray<ImageBuffer,N>& buffers) {
  std::vector<i32> sizes(N);
  for (i32 i = 0; i < N; ++i)
    sizes[i] = buffers.size(i);
  
  std::tr1::shared_ptr< NDArray<ImageData,N> > dataPtr
    (new NDArray<ImageData,N>(&sizes[0]));
  NDArray<ImageData,N>& data = *dataPtr;
  
  i32 numElems = buffers.numElems();
  for (i32 i = 0; i < numElems; ++i)
    data[i] = buffers[i].fetchData();
  
  return dataPtr;
}

template<i32 N>
inline void WriteImageDataToBufferArray(const NDArray<ImageData,N>& data,
                                        NDArray<ImageBuffer,N>& buffers) {
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

namespace detail {
  template<typename T>
  void writeVal(std::ostream& stream, T val, i32 n) {
    while (n--)
      stream.write(reinterpret_cast<char*>(&val), sizeof(T));
  }
  
  template<typename T>
  void writeVals(std::ostream& stream, const T* val, i32 n) {
    stream.write(reinterpret_cast<const char*>(val), n*sizeof(T));
  }
  
  i32 nextAligned(i32 n, i32 align) {
    i32 mod = n%align;
    return n + (mod != 0)*(align - mod);
  }

  void align(std::ostream& stream, i32 align, std::ios::pos_type initialPos) {
    i32 bytesSoFar = stream.tellp() - initialPos;
    writeVal(stream, ' ', nextAligned(bytesSoFar, align) - bytesSoFar);
  }
  
  void writeTag(std::ostream& stream, u32 tag, u32 nBytes) {
    u32 tagElems[] = {tag, nBytes};
    stream.write(reinterpret_cast<char*>(&tagElems), 8);
  }
}

template<i32 N>
inline void WriteMatlabArray(const NDArray<ImageData,N>& data,
                             const std::string& fileName) {
  const i32* sizes = data.sizes();
  i32 nImages = data.numElems();
  i32 width = data[0].width();
  i32 height = data[0].height();
  i32 bytesPerImage = width*height*sizeof(f32);
  
  std::ofstream stream(fileName.c_str(), std::ios::out | std::ios::binary);
  std::ios::pos_type initialPos = stream.tellp();
  
  std::string arrayName("evpout");
  
  // Header
  stream << "MATLAB 5.0 MAT-file, Created by EVP\n";
  detail::align(stream, 124, initialPos);
  
  detail::writeVal(stream, u16(0x0100), 1);
  detail::writeVal(stream, u16(0x4d49), 1);
  
  // Main matrix
  i32 totalSize = 8 + 8 +
    8 + detail::nextAligned((N + 2)*sizeof(i32), 8) +
    8 + detail::nextAligned(arrayName.size(), 8) +
    8 + detail::nextAligned(bytesPerImage*nImages, 8);
  detail::writeTag(stream, 14, totalSize);
  
  // The Array Flags subelement
  detail::writeTag(stream, 6, 8);
  detail::writeVal(stream, u32(7), 1);
  detail::writeVal(stream, ' ', 4);
  
  // The Dimensions Array subelement
  detail::writeTag(stream, 5, (N + 2)*sizeof(i32));
  detail::writeVal(stream, u32(width), 1);
  detail::writeVal(stream, u32(height), 1);
  detail::writeVals(stream, sizes, N);
  detail::align(stream, 8, initialPos);
  
  // The Array Name subelement
  detail::writeTag(stream, 1, arrayName.size());
  stream << arrayName;
  detail::align(stream, 8, initialPos);
  
  // The real part of the matrix
  detail::writeTag(stream, 7, bytesPerImage*nImages);
  for (i32 i = 0; i < nImages; ++i) {
    const ImageDataValues& vals = data[i].data();
    detail::writeVals(stream, &vals[0], vals.size());
  }
  detail::align(stream, 8, initialPos);
  
  stream.close();
}

}

#endif
