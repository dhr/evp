#pragma once
#ifndef EVP_CURVE_LOGLIN_LLBASISSET_H
#define EVP_CURVE_LOGLIN_LLBASISSET_H

#include <algorithm>
#include <vector>

#include "evp/curve/loglin/llbasis.hpp"

namespace evp {
using namespace clip;

class LLBasisSet {
 protected:
  std::vector<LLBasis> bases_;
  i32 kernSize_;
  
  void normalize() {
    i32 maxX = kernSize_/2;
    f64 posSum = 0;
    f64 negSum = 0;
    
    std::vector<LLBasis>::iterator it, end;
    for (i32 x = -maxX; x <= maxX; ++x) {
      f64 val = 0;
      for (it = bases_.begin(), end = bases_.end(); it != end; ++ it)
        val += it->eval(x);
      if (val > 0) posSum += val;
      else negSum -= val;
    }
    
    f64 scale = bases_.size()/std::max(posSum, negSum);
    
    for (it = bases_.begin(), end = bases_.end(); it != end; ++it)
      it->scale *= scale;
  }
  
 public:
  virtual ~LLBasisSet() {}
 
  LLBasisSet() {}
  
  LLBasisSet(std::vector<LLBasis> bases, i32 kernSize)
  : bases_(bases), kernSize_(kernSize) {}
  
  inline i32 kernSize() { return kernSize_; }
  inline i32 numBases() { return i32(bases_.size()); }
  inline LLBasis &operator[](i32 i) { return bases_[i]; }
  
  virtual void combine(f64 deg, bool adapt,
                      InputAdaptor& input,
                      OutputAdaptor& output) = 0;
};

typedef std::tr1::shared_ptr<LLBasisSet> LLBasisSetPtr;

}

#endif
