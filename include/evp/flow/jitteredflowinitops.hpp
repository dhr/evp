#pragma once
#ifndef EVP_FLOW_JITTEREDFLOWINITOPS_H
#define EVP_FLOW_JITTEREDFLOWINITOPS_H

#include <tr1/functional>

#include <clip.hpp>

#include "evp/flow/flowinitops.hpp"
#include "evp/flow/flowtypes.hpp"
#include "evp/util/gabor.hpp"

namespace evp {
using namespace clip;

void LocalNonMaximumSuppression(i32 n, i32 radius,
                                InputAdaptor& input,
                                OutputAdaptor& output) {
  ImBufList circBuf;
  ImBufList temp;
  
  for (i32 i = 0; i < n; i++)
    circBuf.push_back(input.next());
  
  CircularAdaptor circIter(circBuf);
  circIter.advance(radius);
  for (i32 i = 0; i < n; i++) {
    circIter.advance(-2*radius);
    temp.push_front(Merge(Max, 2*radius + 1, circIter));
  }
  
  circIter.reset();
  PopAdaptor popper(temp);
  for (i32 i = 0; i < n; i++) {
    ImageBuffer image = circIter.next();
    output.output(PointwiseThreshold(image, popper.next()));
  }
}

class JitteredFlowInitOps : public FlowInitOps {
  FlowInitOpParams params_;
  NDArray<ImageData,4> filters_;
  f64 power_;
  i32 numOrientationJitters_;
  i32 numScaleJitters_;
  ImageBuffer maxBuf_;
  
 public:
  JitteredFlowInitOps(FlowInitOpParams& params,
                      i32 numOrientationJitters = 1,
                      i32 numScaleJitters = 1)
  : params_(params),
    filters_(params.numOrientations, numOrientationJitters, numScaleJitters, 2),
    numOrientationJitters_(numOrientationJitters),
    numScaleJitters_(numScaleJitters)
  {
    f64 a = 1.5;
    f64 baseWavelength = params.size;
    
    for (i32 ti = 0; ti < params_.numOrientations; ++ti) {
      for (i32 jti = 0; jti < numOrientationJitters_; ++jti) {
        f64 toff = (jti + 0.5)/numOrientationJitters_ - 0.5;
        f64 t = (ti + toff)*params_.orientationStep;
        
        for (i32 jsi = 0; jsi < numScaleJitters_; ++jsi) {
          f64 w = std::pow(2.f, jsi/2.f)*baseWavelength;
          f64 s = w;
          
          filters_(ti, jti, jsi, 0) = MakeGabor(t, w, 0, s, a);
          filters_(ti, jti, jsi, 1) = MakeGabor(t, w, M_PI_2, s, a);
        }
      }
    }
  }
  
  FlowBuffersPtr apply(const ImageBuffer& image) {
    using namespace std;
    using namespace tr1;
    using namespace placeholders;
    
    FlowBuffersPtr outputPtr(new FlowBuffers(params_.numOrientations,
                                             params_.numCurvatures,
                                             params_.numCurvatures));
    FlowBuffers& output = *outputPtr;
    
    ImBufList stack;
    PopAdaptor popper(stack);
    PushAdaptor pusher(stack);

//    GaussianBlurOp blur(6.f);
    ImageBuffer input = image; // blur(image);
    
    f32 mulNorm = 1.f/numScaleJitters_;
    for (i32 ti = 0; ti < params_.numOrientations; ti++) {
      for (i32 jti = 0; jti < numOrientationJitters_; jti++) {
        for (i32 jsi = 0; jsi < numScaleJitters_; jsi++) {
          ImageBuffer o1 = Filter(input, filters_(ti, jti, jsi, 0));
          ImageBuffer o2 = Filter(input, filters_(ti, jti, jsi, 1));
          ImageBuffer energy = Sqrt(o1*o1 + o2*o2);
          pusher.output(energy);
        }
        
        pusher.output(Merge(Mul, numScaleJitters_, popper)^mulNorm);
        
        setProgress(f32(ti*numOrientationJitters_ + jti + 1)/
                    params_.numOrientations/numOrientationJitters_);
      }
    }
    
    i32 n = numOrientationJitters_*params_.numOrientations;
    i32 suppressionRadius = 2*numOrientationJitters_;
    LocalNonMaximumSuppression(n, suppressionRadius, popper, pusher);
    
    for (i32 ti = params_.numOrientations - 1; ti >= 0; ti--) {
      ImageBuffer temp = Merge(Add, numOrientationJitters_, popper);
      Rescale(temp, params_.threshold, 1.f, params_.minConf, 1.f, true, temp);
      
      output(ti, 0, 0) = temp;
      
      for (i32 ktii = 0; ktii < params_.numCurvatures; ++ktii) {
        for (i32 knii = 0; knii < params_.numCurvatures; ++knii) {
          if (!ktii && !knii) continue;
          
          NDIndex<3> target(ti, ktii, knii);
          output[target] = temp.clone();
        }
      }
    }
    
    return outputPtr;
  }
};

}

#endif
