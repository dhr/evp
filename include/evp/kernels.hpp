#pragma once
#ifndef EVP_KERNELS_H
#define EVP_KERNELS_H

#include <clip.hpp>

namespace evp {
namespace detail {

inline const char* LogLinKernels() {
  static const char* kernels =
    #include <clip/kernels/util.clstr>
    #include "kernels/util.clstr"
    #include "kernels/loglin.clstr"
    ;
  return kernels;
}

inline const char* CurveKernels() {
  static const char* kernels =
    #include <clip/kernels/util.clstr>
    #include "kernels/util.clstr"
    #include "kernels/curve.clstr"
    ;
  return kernels;
}

inline const char* FlowKernels() {
  static const char* kernels =
    #include <clip/kernels/util.clstr>
    #include "kernels/util.clstr"
    #include "kernels/flow.clstr"
    ;
  return kernels;
}

}
}

#endif