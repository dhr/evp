#pragma once
#ifndef EVP_FLOW_JITTEREDFLOWINITOPS_H
#define EVP_FLOW_JITTEREDFLOWINITOPS_H

#include <tr1/functional>

#include <clip.hpp>

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

class JitteredFlowInitOps : public Monitorable {
  FlowInitOpParams params_;
  NDArray<ImageData,4> filters_;
  f64 power_;
  i32 numOrientationJitters_;
  i32 numScaleJitters_;
  i32 numOffsets_;
  ImageBuffer maxBuf_;
  
 public:
  JitteredFlowInitOps(FlowInitOpParams &params,
                      i32 numOrientationJitters = 5,
                      i32 numScaleJitters = 5,
                      i32 numOffsets = 5,
                      i32 baseScale = 2)
  : params_(params),
    filters_(params.numOrientations, numOrientationJitters,
             numScaleJitters, numOffsets),
    numOrientationJitters_(numOrientationJitters),
    numScaleJitters_(numScaleJitters),
    numOffsets_(numOffsets)
  {
    f64 baseWavelength = 2*baseScale;
    f64 baseSigma = 2.1*baseScale;
    f64 a = 1.2;
    
    for (i32 ti = 0; ti < params_.numOrientations; ++ti) {
      for (i32 jti = 0; jti < numOrientationJitters_; ++jti) {
        f64 toff = (jti + 0.5)/numOrientationJitters_ - 0.5;
        f64 t = (ti + toff)*params_.orientationStep;
        
        for (i32 jsi = 0; jsi < numScaleJitters_; ++jsi) {
          f64 w = baseWavelength + jsi;//*(1 << jsi);
          f64 s = baseSigma;//*(1 << jsi);
          
          for (i32 jpi = 0; jpi < numOffsets_; ++jpi) {
            f64 p = jpi*(M_PI/2)/(numOffsets_ - 1);
            
            filters_(ti, jti, jsi, jpi) = MakeGabor(t, w, p, s, a);
          }
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

    for (i32 ti = 0; ti < params_.numOrientations; ti++) {
      for (i32 jti = 0; jti < numOrientationJitters_; jti++) {
        for (i32 jsi = 0; jsi < numScaleJitters_; jsi++) {
          for (i32 jpi = 0; jpi < numOffsets_; jpi++) {
            ImageData& filter = filters_(ti, jti, jsi, jpi);
            ImageBuffer temp = Filter(image, filter);
            pusher.output(temp);
            pusher.output(Negate(temp));
          }
        }
        
        i32 n = 2*numScaleJitters_*numOffsets_;
        
        InputIteratorAdaptor<ImBufList::reverse_iterator> iia1(stack.rbegin());
        Map(HalfRectify, n, iia1);
        
        ImageBuffer avg = Merge(Add, n, popper);
        avg /= n/2.f;
        
        pusher.output(avg);
        setProgress(0.9f*f32(ti*numOrientationJitters_ + jti + 1)/
                    params_.numOrientations/
                    numOrientationJitters_);
      }
    }
    
    i32 n = numOrientationJitters_*params_.numOrientations;
    i32 suppressionRadius = 2*numOrientationJitters_;
    LocalNonMaximumSuppression(n, suppressionRadius, popper, pusher);
    
    for (i32 ti = params_.numOrientations - 1; ti >= 0; ti--) {
      ImageBuffer temp = Merge(Add, numOrientationJitters_, popper);
      Bound(temp, temp);
      
      output(ti, 0, 0) = temp;
      
      for (i32 ktii = 0; ktii < params_.numCurvatures; ++ktii) {
        for (i32 knii = 0; knii < params_.numCurvatures; ++knii) {
          if (!ktii && !knii) continue;
          
          NDIndex<3> target(ti, ktii, knii);
          output[target] = temp.clone();
        }
      }
      
      setProgress(0.9f + 0.1f*(1.f - f32(ti)/params_.numOrientations));
    }
    
    return outputPtr;
  }
};

}

#endif
