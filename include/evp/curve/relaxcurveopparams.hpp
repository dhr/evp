#pragma once
#ifndef EVP_CURVE_RELAXCURVEOPPARAMS_H
#define EVP_CURVE_RELAXCURVEOPPARAMS_H

#include "evp/curve/curvefeaturetypes.hpp"

namespace evp {
using namespace clip;

class RelaxCurveOpParams {
  FeatureType featureType;
  i32 numOrientations;
  i32 numCurvatures;
  f32 curvatureStep;
  i32 kernelSize;
  i32 numTangentialComponents;
  f32 stabilizer;
  f32 stabilizerDegree;
  f32 sigmaXY;
  f32 sigmaTheta;
  f32 sigmaKappa;
  f32 sigmaTransport;
  f32 dilation;
  f32 offset;
  f32 logLinDegree;
  f32 adapt;
  
  // Inferred parameters
  
  f32 orientationStep;
  i32 numCurvClasses;
  i32 numPis;
  i32 numTotalOrientations;
  i32 numNormalComponents;
  
  friend class RelaxCurveOp;
  friend class CurveSupportOp;
  friend class CurveCompatKern;
  
 public:
  RelaxCurveOpParams(FeatureType ft, i32 nt = 8, i32 nk = 5)
  : featureType(ft),
    numOrientations(nt), numCurvatures(nk),
    curvatureStep(0.1), kernelSize(19),
    numTangentialComponents(4), stabilizer(0.55),
    stabilizerDegree(2),
    sigmaXY(sqrt(2)/2), sigmaTheta(sqrt(2)/2),
    sigmaKappa(0.75), sigmaTransport(3),
    dilation(0.1), offset(sqrt(2)/2),
    logLinDegree(4), adapt(true),
    orientationStep(M_PI/nt), numCurvClasses(nk/2 + 1),
    numPis(ft == Lines ? 1 : 2),
    numTotalOrientations((ft == Lines ? 1 : 2)*nt),
    numNormalComponents(nk > 1 ? 6 : 4) {}
  
  FeatureType feature() { return featureType; }
  i32 orientationsPerPi() { return numOrientations; }
  bool piSymmetric() { return featureType == Lines; }
  i32 curvatures() { return numCurvatures; }
};

}

#endif
