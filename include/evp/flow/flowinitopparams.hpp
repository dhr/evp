#pragma once
#ifndef EVP_FLOW_FLOWINITOPPARAMS_H
#define EVP_FLOW_FLOWINITOPPARAMS_H

namespace evp {
using namespace clip;

struct FlowInitOpParams {
  i32 numOrientations;
  i32 numCurvatures;
  f32 curvatureStep;
  f32 size;
  f32 threshold;
  f32 minConf;
  bool estimateCurvatures;
  
  // Inferred parameters
  
  f32 orientationStep;
  i32 numPis;
  i32 numCurvClasses;
  
  FlowInitOpParams(i32 nt = 8, i32 nk = 5)
  : numOrientations(nt),
    numCurvatures(nk),
    curvatureStep(0.1),
    size(2.f),
    threshold(0.0),
    minConf(0.7f),
    estimateCurvatures(false),
    orientationStep(M_PI/nt),
    numPis(1),
    numCurvClasses(nk/2 + 1) {}
};

}

#endif
