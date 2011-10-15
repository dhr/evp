#pragma once
#ifndef EVP_KERNELS_H
#define EVP_KERNELS_H

#include <clip.hpp>

namespace evp {
namespace detail {

inline const char* LogLinKernels() {
  static const char* kernels =
    #include <clip/kernels/util.cl>
    #include "kernels/util.cl"
    #include "kernels/loglin.cl"
    ;
  return kernels;
}

inline const char* CurveKernels() {
  static const char* kernels =
    #include <clip/kernels/util.cl>
    #include "kernels/util.cl"
    #include "kernels/curve.cl"
    ;
  return kernels;
}

inline const char* FlowKernels() {
  static const char* kernels =
    #include <clip/kernels/util.cl>
    #include "kernels/util.cl"
    #include "kernels/flow.cl"
    ;
  return kernels;
}

}
}

#endif