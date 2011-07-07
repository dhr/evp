#pragma once
#ifndef EVP_CURVE_LOGLIN_LLENDSTOPPEDOP_H
#define EVP_CURVE_LOGLIN_LLENDSTOPPEDOP_H

#include <cmath>

#include "memory.h"

#include <clip.hpp>

#include "evp/curve/loglin/llcellop.hpp"
#include "evp/curve/loglin/llsimplecellop.hpp"

namespace evp {
using namespace clip;

class LLEndstoppedOp : public LLCellOp {
  LLSimpleCellOpPtr excComponent_;
  LLSimpleCellOpPtr posKInhComponent_;
  LLSimpleCellOpPtr negKInhComponent_;
  
 public:
  template<typename T>
  LLEndstoppedOp(T excComponent,
                 T posKInhComponent,
                 T negKInhComponent)
  : excComponent_(excComponent),
    posKInhComponent_(posKInhComponent),
    negKInhComponent_(negKInhComponent) {}
  
  void apply(const ImageBuffer& image, ImageBuffer output[4]) {
    ImageBuffer temps[6];
    excComponent_->apply(image, temps);
    posKInhComponent_->apply(image, temps + 2);
    negKInhComponent_->apply(image, temps + 4);
    for (i32 i = 0; i < 6; i++)
      HalfRectify(temps[i], temps[i]);
    output[0] = temps[0] - temps[2];
    output[1] = temps[1] - temps[3];
    output[2] = temps[0] - temps[4];
    output[3] = temps[1] - temps[5];
  }
};

}

#endif
