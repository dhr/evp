#pragma once
#ifndef EVP_FLOW_GRADIENTFLOWINITOPS_H
#define EVP_FLOW_GRADIENTFLOWINITOPS_H

#include <clip.hpp>

#include "evp/flow/flowinitops.hpp"
#include "evp/flow/flowtypes.hpp"

namespace evp {
using namespace clip;

struct KtKnOp : public BasicOp {
  KtKnOp() : BasicOp("ktkn") {}
  
  template<typename T>
  void operator()(const T& us, const T& vs, const T& vxs, const T& vys,
                  T kts, T kns) {
    cl::Kernel& kernel = cache_.get();
    kernel.setArg(0, us.mem());
    kernel.setArg(1, vs.mem());
    kernel.setArg(2, vxs.mem());
    kernel.setArg(3, vys.mem());
    kernel.setArg(4, kts.mem());
    kernel.setArg(5, kns.mem());
    Enqueue(kernel, kts);
  }
};
KtKnOp KtKn;

struct DiscretizeFlowOp : public BasicOp {
  DiscretizeFlowOp() : BasicOp("flowdiscr") {}

  template<typename T>
  T operator()(const T& confs, const T& thetas,
               f32 targTheta, f32 thetaStep, i32 numPis,
               const T& kts, const T& kns,
               f32 targKt, f32 targKn, f32 kStep,
               T o) {
    cl::Kernel& kernel = cache_.get();
    kernel.setArg(0, confs.mem());
    kernel.setArg(1, thetas.mem());
    kernel.setArg(2, targTheta);
    kernel.setArg(3, thetaStep);
    kernel.setArg(4, numPis);
    kernel.setArg(5, kts.mem());
    kernel.setArg(6, kns.mem());
    kernel.setArg(7, targKt);
    kernel.setArg(8, targKn);
    kernel.setArg(9, kStep);
    kernel.setArg(10, o.mem());
    Enqueue(kernel, o);
    return o;
  }
};
DiscretizeFlowOp DiscretizeFlow;

class GradientFlowInitOps : public FlowInitOps {
  FlowInitOpParams &params_;
  GradientOp gradient_;
  GaussianBlurOp blurImage_;
  GaussianBlurOp blurUV_;
  GaussianBlurOp blurVGrad_;
  
 public:
  GradientFlowInitOps(FlowInitOpParams &params)
  : params_(params),
    blurImage_(params.size),
    blurUV_(params.size),
    blurVGrad_(params.size) {}
  
  FlowBuffersPtr apply(const ImageBuffer& image) {
    FlowBuffersPtr outputPtr(new FlowBuffers(params_.numOrientations,
                                             params_.numCurvatures,
                                             params_.numCurvatures));
    FlowBuffers& output = *outputPtr;
    
    ImageBuffer gradX = ~image, gradY = ~image;
    if (params_.size > 0)
      gradient_(blurImage_(image), gradX, gradY);
    else
      gradient_(image, gradX, gradY);
    
    ImageBuffer thetas = ~image, confs = ~image;
    Grad2Polar(gradX, gradY, confs, thetas);

    confs /= MaxReduce(confs);
    Rescale(confs, params_.threshold, 1.f,
            params_.minConf, 1.f, true,
            confs);
    
    setProgress(0.1f);
    
    ImageBuffer kts = ~image, kns = ~image;
    if (params_.estimateCurvatures) {
      ImageBuffer us = ~image, vs = ~image;
      UnitVectorize(thetas, us, vs);
      
      if (params_.size > 0) {
        blurUV_(us, us);
        blurUV_(vs, vs);
      }
      
      ImageBuffer vxs = ~image, vys = ~image;
      gradient_(vs, vxs, vys);
      
      if (params_.size > 0) {
        blurVGrad_(vxs, vxs);
        blurVGrad_(vys, vys);
      }
      
      kts = ~image;
      kns = ~image;
      
      KtKn(us, vs, vxs, vys, kts, kns);
    }
    else {
      Memset(kts, 0);
      Memset(kns, 0);
    }
    
    NDIndex<3> index;
    for (i32 ti = 0; ti < params_.numOrientations; ++ti) {
      index[0] = ti;
      
      setProgress(0.2f + 0.8f*f32(ti)/params_.numOrientations);
      
      for (i32 kti = 0; kti < params_.numCurvatures; ++kti) {
        index[1] = kti;
        
        for (i32 kni = 0; kni < params_.numCurvatures; ++kni) {
          index[2] = kni;
          
          f32 theta = ti*params_.orientationStep;
          f32 kt = params_.estimateCurvatures*kti*params_.curvatureStep;
          f32 kn = params_.estimateCurvatures*kni*params_.curvatureStep;
          
          output[index] = ~image;
          DiscretizeFlow(confs, thetas, theta,
                         params_.orientationStep, params_.numPis,
                         kts, kns, kt, kn, params_.curvatureStep,
                         output[index]);
        }
      }
    }
    
    setProgress(1.f);
    
    return outputPtr;
  }
};

}

#endif
