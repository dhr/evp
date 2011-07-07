#pragma once
#ifndef EVP_FLOW_FLOWINITOPPARAMS_H
#define EVP_FLOW_FLOWINITOPPARAMS_H

namespace evp {
using namespace clip;

struct FlowInitOpParams {
  i32 numOrientations;
  i32 numCurvatures;
  f32 curvatureStep;
  bool estimateCurvatures;
  f32 blurImageSigma;
  f32 blurUVSigma;
  f32 blurVGradSigma;
  f32 thetaThreshold;
  f32 minConf;
  
  // Inferred parameters
  
  f32 orientationStep;
  i32 numPis;
  i32 numCurvClasses;
  
  FlowInitOpParams(i32 nt = 8, i32 nk = 5)
  : numOrientations(nt),
    numCurvatures(nk),
    curvatureStep(0.1),
    estimateCurvatures(false),
    blurImageSigma(2.f),
    blurUVSigma(1.f),
    blurVGradSigma(1.f),
    thetaThreshold(0.0),
    minConf(0.8),
    orientationStep(M_PI/nt),
    numPis(1),
    numCurvClasses(nk/2 + 1) {}
};

}

#endif
