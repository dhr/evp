#pragma once
#ifndef EVP_CURVE_LOGLIN_LLBASIS_H
#define EVP_CURVE_LOGLIN_LLBASIS_H

#include "evp/util/memutil.hpp"

namespace evp {
using namespace clip;

struct SimpleFunc { virtual f64 operator()(f64 x) = 0; };

typedef std::tr1::shared_ptr<SimpleFunc> SimpleFuncPtr;

class LLBasis {
  SimpleFuncPtr f_;
  i32 kernSize_;
  f64 offset_;
  bool balanced_;
  
  inline f64 eval_(f64 x) {
    return (*f_)(x - offset_);
  }
  
 public:
  f64 scale;
  
  virtual ~LLBasis() {}
  
  LLBasis() {}
  
  LLBasis(SimpleFunc *f, i32 kernSize,
          f64 scl, f64 offset, bool balanced = true,
          bool norm = true)
  : f_(f), kernSize_(kernSize), offset_(offset), balanced_(balanced),
    scale(scl)
  {
    if (norm) normalize();
  }
  
  void normalize() {
    i32 max = kernSize_/2;
    f64 posSum = 0;
    f64 negSum = 0;
    for (i32 x = -max; x <= max; x++) {
      f64 val = eval(x);
      if (val > 0) posSum += val;
      else negSum += val;
    }
    
    if (balanced_) {
      scale /= posSum;
    } else {
      scale /= posSum + negSum;
    }
  }
  
  inline i32 kernSize() { return kernSize_; }
  
  inline bool isBalanced() { return balanced_; }
  
  inline f64 eval(f64 x) { return scale*eval_(x); }
};

}

#endif
