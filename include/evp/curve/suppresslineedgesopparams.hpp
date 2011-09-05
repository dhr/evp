#pragma once
#ifndef EVP_CURVE_EDGESUPPRESSOPPARAMS_H
#define EVP_CURVE_EDGESUPPRESSOPPARAMS_H

#include "evp/curve/curvefeaturetypes.hpp"

namespace evp {
using namespace clip;

class SuppressLineEdgesOpParams {
  i32 numOrientations;
  i32 numCurvatures;
  f32 curvatureStep;
  i32 kernelSize;
  f32 sigmaXY;
  f32 sigmaTheta;
  f32 sigmaTransport;
  
  // Inferred parameters
  
  f32 orientationStep;
  i32 numCurvClasses;
  i32 numTotalOrientations;
  
  friend class LineEdgeLocatorOp;
  friend class SuppressLineEdgesOp;
  
 public:
  SuppressLineEdgesOpParams(i32 nt = 8, i32 nk = 5)
  : numOrientations(nt), numCurvatures(nk),
    curvatureStep(0.1), kernelSize(19),
    sigmaXY(sqrt(2)), sigmaTheta(sqrt(2)*M_PI/nt),
    sigmaTransport(sqrt(2)/2),
    orientationStep(M_PI/nt), numCurvClasses(nk/2 + 1),
    numTotalOrientations(2*nt) {}
  
  i32 orientationsPerPi() { return numOrientations; }
  i32 curvatures() { return numCurvatures; }
};

}

#endif
