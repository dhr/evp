#pragma once
#ifndef EVP_CURVE_RELAXCURVEOP_H
#define EVP_CURVE_RELAXCURVEOP_H

#include <cmath>

#include <clip.hpp>

#include "evp/curve/curvetypes.hpp"
#include "evp/curve/curvesupportop.hpp"
#include "evp/curve/relaxcurveopparams.hpp"
#include "evp/util/memutil.hpp"
#include "evp/util/monitorable.hpp"

namespace evp {
using namespace clip;

class RelaxCurveOp : public Monitorable {    
 private:
  RelaxCurveOpParams params_;
  NDArray<CurveSupportOpPtr,2> ops_;
  f32 relaxStep_;
  i32 iterations_;
  f32 threshold_;
  
 public:
  RelaxCurveOp(RelaxCurveOpParams &ps,
               i32 iters = 5,
               f32 relaxStep = 1.f,
               f32 threshold = 0.f)
  : params_(ps), ops_(ps.numTotalOrientations, ps.numCurvatures),
    relaxStep_(relaxStep), iterations_(iters), threshold_(threshold)
  {
    for (i32 tii = 0; tii < params_.numTotalOrientations; ++tii) {
      f64 ti = tii*params_.orientationStep;
      
      for (i32 kii = 0; kii < params_.numCurvatures; ++kii) {
        f64 ki =
          (i32(kii) + 1 - i32(params_.numCurvClasses))*params_.curvatureStep;
        
        ops_(tii, kii) =
          CurveSupportOpPtr(new CurveSupportOp(ti, ki, params_));
      }
    }
  }
  
  CurveBuffersPtr apply(const CurveBuffers& input) {
    i32 nt = params_.numOrientations;
    i32 nk = params_.numCurvatures;
    
    if (input.size(0) == 2*nt) nt *= 2;
    
    assert(nt == input.size(0) && nk == input.size(1) &&
           "Input has incompatible size for current parameters.");
    
    CurveBuffersPtr outputPtr1(new CurveBuffers(nt, nk));
    CurveBuffersPtr outputPtr2(new CurveBuffers(nt, nk));
    
    CurveBuffersPtr outputPtr = outputPtr1, temp = outputPtr2;
    
    NDIndex<2> index;
    for (i32 iter = 0; iter < iterations_; iter++) {
      const CurveBuffers& relaxSrc = iter == 0 ? input : *outputPtr;
      
      for (i32 ti = 0; ti < nt; ti++) {
        index[0] = ti;
        
        for (i32 ki = 0; ki < nk; ki++) {
          index[1] = ki;
          
          ImageBuffer support = ops_(ti%ops_.size(0), ki)->apply(relaxSrc);
          MulAdd(relaxSrc[index], support, relaxStep_, support);
          if (threshold_ > 0)
            Threshold(support, threshold_, support);
          (*temp)[index] = Bound(support, support);
          setProgress(f32(iter*nt*nk + ti*nk + ki + 1)/(iterations_*nt*nk));
        }
      }
      
      std::swap(outputPtr, temp);
    }
    
    return outputPtr;
  }
  
  inline f32 relaxationDelta() { return relaxStep_; }
  inline void setRelaxationDelta(f32 delta) { relaxStep_ = delta; }
  
  inline f32 iterations() { return iterations_; }
  inline void setIterations(i32 iterations) { iterations_ = iterations; }
};

}

#endif
