#pragma once
#ifndef EVP_CURVE_LOGLIN_LLINITOPKERN_H
#define EVP_CURVE_LOGLIN_LLINITOPKERN_H

#include "evp/util/mathutil.hpp"
#include "evp/curve/loglin/llbasis.hpp"

namespace evp {
using namespace clip;

inline ImageData MakeLLInitOpKern(f64 orientation,
                                  LLBasis &nbasis,
                                  LLBasis &tbasis,
                                  i32 kernSize) {
  i32 halfKernSize = kernSize/2;
  kernSize = 2*halfKernSize + 1;
  ImageData img(kernSize, kernSize);
  
  f64 cosDir = cos(orientation);
  f64 sinDir = sin(orientation);
  
  for (i32 y = -halfKernSize, i = 0; y <= halfKernSize; ++y, ++i) {
    for (i32 x = -halfKernSize, j = 0; x <= halfKernSize; ++x, ++j) {
      img(j, i) = (tbasis.eval(cosDir*x + sinDir*y)*
                   nbasis.eval(cosDir*y - sinDir*x));
    }
  }
  
  if (nbasis.isBalanced() || tbasis.isBalanced())
    img.balance();
  
  return img;
};

}

#endif
