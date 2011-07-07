#pragma once
#ifndef EVP_CURVE_LOGLIN_LLSIMPLECELLOP_H
#define EVP_CURVE_LOGLIN_LLSIMPLECELLOP_H

#include <cmath>

#include <algorithm>
#include <list>
#include "memory.h"
#include <vector>

#include <clip.hpp>

#include "evp/curve/loglin/llcellop.hpp"
#include "evp/curve/loglin/llinitopkern.hpp"
#include "evp/util/ndarray.hpp"

namespace evp {
using namespace clip;

struct LLSimpleCellOp : LLCellOp {
  f32 orientation_;
  LLBasisSetPtr tangentialBases_;
  LLBasisSetPtr normalBases_;
  NDArray<SparseImageData,2> filters_;
  f32 scale_;
  f32 deg_;
  bool adapt_;
  
  LLSimpleCellOp(f64 theta,
                 LLBasisSetPtr normalBases, LLBasisSetPtr tangentialBases,
                 f64 scale, f64 deg, bool adapt = false)
  : orientation_(theta),
    tangentialBases_(tangentialBases), normalBases_(normalBases),
    filters_(tangentialBases->numBases(), normalBases->numBases()),
    scale_(scale), deg_(deg), adapt_(adapt)
  {
    for (i32 ti = 0; ti < tangentialBases->numBases(); ++ti) {
      LLBasis& tbasis = (*tangentialBases)[ti];
      
      for (i32 ni = 0; ni < normalBases->numBases(); ++ni) {
        LLBasis& nbasis = (*normalBases)[ni];
        
        i32 kernSize = std::max(tbasis.kernSize(), nbasis.kernSize());
        ImageData opKern = MakeLLInitOpKern(theta, nbasis, tbasis, kernSize);
        filters_(ti, ni) = SparseImageData(opKern);
      }
    }
  }
  
  void apply(const ImageBuffer& image, ImageBuffer output[2]) {   
    ImBufList stack[2];
    PopAdaptor popper[2] = {PopAdaptor(stack[0]), PopAdaptor(stack[1])};
    PushAdaptor pusher[2] = {PushAdaptor(stack[0]), PushAdaptor(stack[1])};
    
    for (i32 ti = 0; ti < filters_.size(0); ++ti) {
      for (i32 ni = 0; ni < filters_.size(1); ++ni) {
        ImageBuffer filtered = Filter(image, filters_(ti, ni));
        filtered *= scale_;
        pusher[0].output(filtered);
        pusher[1].output(Negate(popper[0].peek()));
      }
      
      normalBases_->combine(deg_/scale_, adapt_, popper[0], pusher[0]);
      normalBases_->combine(deg_/scale_, adapt_, popper[1], pusher[1]);
    }
    
    tangentialBases_->combine(deg_/scale_, adapt_, popper[0], pusher[0]);
    tangentialBases_->combine(deg_/scale_, adapt_, popper[1], pusher[1]);
    
    output[0] = popper[0].next();
    output[1] = popper[1].next();
  }
};

typedef std::tr1::shared_ptr<LLSimpleCellOp> LLSimpleCellOpPtr;

}

#endif
