#pragma once
#ifndef EVP_IO_MATLABIO_H
#define EVP_IO_MATLABIO_H

#ifndef EVP_NO_MATIO
# include <matio.h>
#endif

#include <clip.hpp>

#include "evp/util/memutil.hpp"
#include "evp/util/ndarray.hpp"

namespace evp {
using namespace clip;

#ifndef EVP_NO_MATIO
template<i32 N>
inline bool WriteMatlabArray(const std::string& fileName,
                             const NDArray<ImageData,N>& data) {
  mat_t* matfp = Mat_CreateVer(fileName.c_str(), NULL, MAT_FT_MAT5);
  if (!matfp) {
    std::cerr << "Error writing MAT file " << fileName << std::endl;
    return false;
  }
  
  i32 nImages = data.numElems();
  i32 elemsPerImage = data[0].width();
  elemsPerImage *= data[0].height();
  
  std::vector<size_t> sizes(N + 2);
  sizes[0] = data[0].width();
  sizes[1] = data[0].height();
  for (i32 i = 0; i < N; ++i) sizes[i + 2] = data.size(i);
  
  std::vector<float> elems(nImages*elemsPerImage);
  for (i32 i = 0; i < nImages; ++i) {
    memcpy(&elems[i*elemsPerImage], &data[i][0], elemsPerImage*sizeof(float));
  }
  
  matvar_t* matvar = Mat_VarCreate("evpdata", MAT_C_SINGLE, MAT_T_SINGLE,
                                   N + 2, &sizes[0], &elems[0], 0);
  if (!matvar) {
    std::cerr << "Error creating MATLAB output variable" << std::endl;
    Mat_Close(matfp);
    return false;
  }
  
  Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_ZLIB);
  Mat_VarFree(matvar);
  
  Mat_Close(matfp);
  
  return true;
}

template<i32 N>
std::tr1::shared_ptr<NDArray<ImageData,N> >
ReadMatlabArray(const std::string& fileName) {
  std::tr1::shared_ptr<NDArray<ImageData,N> > output;
  
  mat_t* matfp = Mat_Open(fileName.c_str(), MAT_ACC_RDONLY);
  if (!matfp) {
    std::cerr << "Error reading MAT file " << fileName << std::endl;
    return output;
  }
  
  matvar_t* matvar = Mat_VarRead(matfp, "evpdata");
  if (!matvar) {
    std::cerr << "Couldn't find the 'evpdata' variable" << std::endl;
  }
  else {
    if (matvar->class_type != MAT_C_SINGLE) {
      std::cerr << "The 'evpdata' variable isn't type 'single'" << std::endl;
    }
    else if (matvar->rank != N + 2) {
      std::cerr << "The 'evpdata' variable has rank " << matvar->rank
                << ", expected " << N + 2 << std::endl;
    }
    else {
      output =
        std::tr1::shared_ptr<NDArray<ImageData,N> >
          (new NDArray<ImageData,N>(matvar->dims + 2));
      f32* vardata = static_cast<f32*>(matvar->data);
      i32 w = matvar->dims[0], h = matvar->dims[1];
      i32 n = w*h;
      
      i32 nImages = 1;
      for (i32 i = 0; i < N; ++i)
        nImages *= matvar->dims[i + 2];
      
      for (i32 i = 0; i < nImages; ++i) {
        (*output)[i] = ImageData(w, h);
        memcpy(&(*output)[i][0], vardata + i*n, n*sizeof(float));
      }
    }
    
    Mat_VarFree(matvar);
  }
  
  Mat_Close(matfp);
  
  return output;
}
#else
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
inline bool WriteMatlabArray(const std::string& fileName,
                             const NDArray<ImageData,N>& data) {
  const i32* sizes = data.sizes();
  i32 nImages = data.numElems();
  i32 width = data[0].width();
  i32 height = data[0].height();
  i32 bytesPerImage = width*height*sizeof(f32);
  
  std::ofstream stream(fileName.c_str(), std::ios::out | std::ios::binary);
  if (stream.fail())
    throw std::runtime_error("Failed to open file " + fileName);
  
  std::ios::pos_type initialPos = stream.tellp();
  
  std::string arrayName("evpdata");
  
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
#endif
  
}

#endif
