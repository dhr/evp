#pragma once
#ifndef EVP_FLOW_PUSHPULLINITOPS_H
#define EVP_FLOW_PUSHPULLINITOPS_H

#include <tr1/functional>

#include <clip.hpp>

#include "evp/flow/flowinitops.hpp"
#include "evp/flow/flowtypes.hpp"
#include "evp/util/gabor.hpp"

namespace evp {
using namespace clip;

class PushPullInitOps : public FlowInitOps {
  FlowInitOpParams params_;
  ImageData dogFilter_;
  NDArray<SparseImageData,5> filters_;
  i32 numOrientationJitters_;
  i32 numScaleJitters_;
  ImageBuffer maxBuf_;
  
 public:
  PushPullInitOps(FlowInitOpParams& params,
                  i32 numOrientationJitters = 1,
                  i32 numScaleJitters = 1)
  : params_(params),
    filters_(params.numOrientations, numOrientationJitters, numScaleJitters, 2),
    numOrientationJitters_(numOrientationJitters),
    numScaleJitters_(numScaleJitters)
  {
    f64 a = 1.5;
    f64 baseWavelength = params_.size;
    
    f64 dogSigma = 2.5*params_.size;
    i32 dogWidth = i32(floor(6*dogSigma)) | 1;
    dogFilter_ = ImageData(dogWidth, dogWidth);
    
    for (i32 y = 0; y < dogFilter_.height(); ++y) {
      i32 offy = y - dogFilter_.height()/2;
      
      for (i32 x = 0; x < dogFilter_.width(); ++x) {
        i32 offx = x - dogFilter_.width()/2;
        
        f32 d = sqrt(offx*offx + offy*offy);
        dogFilter_(x, y) = Gaussian(d, dogSigma/2) - Gaussian(d, dogSigma);
      }
    }
    
    dogFilter_.balance();
    dogFilter_.normalize();
    
    for (i32 ti = 0; ti < params_.numOrientations; ++ti) {
      for (i32 jti = 0; jti < numOrientationJitters_; ++jti) {
        f64 toff = (jti + 0.5)/numOrientationJitters_ - 0.5;
        f64 t = (ti + toff)*params_.orientationStep;
        
        for (i32 jsi = 0; jsi < numScaleJitters_; ++jsi) {
          f64 w = std::pow(2.f, jsi/2.f)*baseWavelength;
          f64 s = w;
          
          for (i32 phase = 0; phase <= 1; ++phase) {
            ImageData posGabor = MakeGabor(t, w, phase*M_PI_2, s, a);
            posGabor.data() /= std::abs(posGabor.data()).max();
            
            ImageData negGabor = posGabor.clone();
            negGabor.data() *= -1;
            
            for (i32 y = 0; y < posGabor.height(); ++y) {
              for (i32 x = 0; x < posGabor.width(); ++x) {
                if (posGabor(x, y) >= 0)
                  negGabor(x, y) = 0;
                else
                  posGabor(x, y) = 0;
              }
            }
            
            filters_(ti, jti, jsi, phase, 0) = SparseImageData(posGabor);
            filters_(ti, jti, jsi, phase, 1) = SparseImageData(negGabor);
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

//    GaussianBlurOp blur(6.f);
    ImageBuffer input = Filter(image, dogFilter_); // blur(image);
    ImageBuffer posInput = Max(input, 0.f);
    ImageBuffer negInput = -Min(input, 0.f);
    
    f32 mulNorm = 1.f/numScaleJitters_;
    for (i32 ti = 0; ti < params_.numOrientations; ti++) {
      for (i32 jti = 0; jti < numOrientationJitters_; jti++) {
        for (i32 jsi = 0; jsi < numScaleJitters_; jsi++) {
          ImageBuffer phases[2];
          
          for (i32 phase = 0; phase <= 1; ++phase) {
            ImageBuffer inPhase =
              Filter(posInput, filters_(ti, jti, jsi, phase, 0)) +
              Filter(negInput, filters_(ti, jti, jsi, phase, 1));
            ImageBuffer antiPhase =
              Filter(negInput, filters_(ti, jti, jsi, phase, 0)) +
              Filter(posInput, filters_(ti, jti, jsi, phase, 1));
            
            phases[phase] = Max(inPhase - 1.01*antiPhase, 0);
          }
          
          ImageBuffer energy = phases[0]; // Sqrt((phases[0]^2) + (phases[1]^2));
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
