#pragma once
#ifndef EVP_CURVE_LOGLIN_LLINITOPPARAMS_H
#define EVP_CURVE_LOGLIN_LLINITOPPARAMS_H

namespace evp {
using namespace clip;

class LLInitOpParams {
  class CurvatureParams {
   public:
    CurvatureParams() {}
    
    CurvatureParams(f32 es, f32 is,
                    f32 ets, f32 its,
                    i32 ec, i32 ic) :
    excScaling(es), inhScaling(is),
    excTangentialSigma(ets), inhTangentialSigma(its),
    excComponents(ec), inhComponents(ic) {}
    
    f32 excScaling;
    f32 inhScaling;
    f32 excTangentialSigma;
    f32 inhTangentialSigma;
    i32 excComponents;
    i32 inhComponents;
  };
  
  FeatureType featureType;
  i32 numOrientations;
  i32 numCurvatures;
  f32 scale;
  
  std::vector<CurvatureParams> curvatureParams;
  f32 stabilizerDegree;
  f32 excStabilizer;
  f32 inhStabilizer;
  f32 normalSigma;
  f32 normalOffset;
  f32 logLinDegree;
  bool adapt;
  
  // Inferred parameters
  
  f32 orientationStep;
  i32 numCurvClasses;
  
  void scaleBy(f32 scl) {
  }
  
  friend class LLInitOps;
  
 public:
  LLInitOpParams(FeatureType ft,
                 i32 nt = 8,
                 i32 nk = 5,
                 f32 scl = 1)
  : featureType(ft),
    numOrientations(nt), numCurvatures(nk),
    curvatureParams(std::vector<CurvatureParams>(nk/2 + 1)),
    stabilizerDegree(2), excStabilizer(0.2), inhStabilizer(0),
    normalSigma(sqrt(2.f)), normalOffset(sqrt(2.f)/2),
    logLinDegree(16), adapt(false),
    orientationStep(M_PI/nt), numCurvClasses(nk/2 + 1)
  {
    for (i32 i = 0; i < numCurvClasses; i++) {
      switch (i) {
        case 0: {
          f32 excScl = (featureType == Lines ? 1.1 : 1.5);
          curvatureParams[i] =
          CurvatureParams(excScl, 0, 2.8, 0, 4, 0);
          break;
        }
        case 1: {
          f32 excScl = featureType == Lines ? 1.2 : 1.6;
          f32 inhScl = featureType == Lines ? 2.2 : 3;
          curvatureParams[i] =
          CurvatureParams(excScl, inhScl, 2.4, 3.2, 4, 4);
          break;
        }
        case 2: {
          f32 excScl = featureType == Lines ? 1.3 : 1.75;
          f32 inhScl = featureType == Lines ? 2.8 : 3.5;
          curvatureParams[i] =
          CurvatureParams(excScl, inhScl, 1.67, 2.3, 2, 4);
          break;
        }
        default:
          break;
      }
    }
    
    scale = scl;
    for (i32 i = 0; i < numCurvClasses; i++) {
      curvatureParams[i].excTangentialSigma *= scl;
      curvatureParams[i].inhTangentialSigma *= scl;
    }
    normalSigma *= scl;
    normalOffset *= scl;
  }
  
  FeatureType feature() { return featureType; }
  i32 orientationsPerPi() { return numOrientations; }
  i32 curvatures() { return numCurvatures; }
  f32 scalingFactor() { return scale; }
};

}

#endif
