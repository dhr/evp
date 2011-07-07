#pragma once
#ifndef EVP_CURVE_LOGLIN_LLSUPPORTOPS_H
#define EVP_CURVE_LOGLIN_LLSUPPORTOPS_H

#include <clip.hpp>

#include "evp/curve/loglin/llbufferops.hpp"

namespace evp {
using namespace clip;

inline void RotateList(clip::ImBufList& list, i32 amount) {
  if (amount > 0) {
    for (i32 j = 0; j < amount; ++j) {
      list.push_front(list.back());
      list.pop_back();
    }
  }
  else {
    for (i32 j = 0; j < -amount; ++j) {
      list.push_back(list.front());
      list.pop_front();
    }
  }
}

inline ImageBuffer TangentialCombine(i32 n, f32 degree, bool adapt,
                                     InputAdaptor& input, ImageBuffer output) {
  using namespace clip;

  i32 extLen = n/2 - 1;
  
  if (n > 2) {
    ImBufList list;
    PopAdaptor popper(list);
    PushAdaptor pusher(list);
    
    PopulateListFromInput(input, n, list);
    
    if (extLen > 1)
      pusher.output(LLMerge(LLAnd, extLen, degree, adapt, 1.f/extLen, popper));
    
    RotateList(list, 1);
    pusher.output(LLMerge(LLAnd, 2, degree, adapt, 1.f/2, popper));
    RotateList(list, 1);
    
    if (extLen > 1)
      pusher.output(LLMerge(LLAnd, extLen, degree, adapt, 1.f/extLen, popper));
    
    pusher.output(LLMerge(LLOr, 2, degree, adapt, 1.f/2, popper));
    
    return LLMerge(LLAnd, 2, degree, adapt, 1.f/2, popper, output);
  }
  
  return LLAnd(input.next(), input.next(), degree, adapt, 1.f/2, output);
}

inline ImageBuffer TangentialCombine(i32 n, f32 degree, bool adapt,
                                     InputAdaptor& input) {
  return TangentialCombine(n, degree, adapt, input, ~input.peek());
}

struct StabilizeOp : public BasicOp {
  typedef ImageBuffer result_type;

  StabilizeOp() : BasicOp("stabilize") {}
  
 ImageBuffer operator()(const ImageBuffer& i1,
                               const ImageBuffer& sum,
                               i32 n, f32 stabAmt,
                              ImageBuffer output) {
    cl::Kernel& kernel = cache_.get();
    
    i32 i = 0;
    kernel.setArg(i++, i1.mem());
    kernel.setArg(i++, sum.mem());
    kernel.setArg(i++, n);
    kernel.setArg(i++, stabAmt);
    kernel.setArg(i++, output.mem());
    
    return output;
  }
  
 ImageBuffer operator()(const ImageBuffer& i1,
                               const ImageBuffer& sum,
                               i32 n, f32 stabAmt) {
    return operator()(i1, sum, n, stabAmt, ~i1);
  }
};
static StabilizeOp Stabilize;

struct SurroundOp : public BasicOp {
  typedef ImageBuffer result_type;
  
  SurroundOp() : BasicOp("surround2") {}
  
  ImageBuffer operator()(const ImageBuffer& i1,
                         const ImageBuffer& i2,
                         f32 degree,
                        ImageBuffer output) {
    cl::Kernel& kernel = cache_.get();
    kernel.setArg(0, i1.mem());
    kernel.setArg(1, i2.mem());
    kernel.setArg(2, degree);
    kernel.setArg(3, output.mem());
    
    Enqueue(kernel, output);
    
    return output;
  }
  
 ImageBuffer operator()(const ImageBuffer& i1,
                               const ImageBuffer& i2,
                               f32 degree) {
    return operator()(i1, i2, degree, ~i1);
  }
};
static SurroundOp Surround;

}

#endif
