#pragma once
#ifndef EVP_FLOW_FLOWCOMPATOPS_H
#define EVP_FLOW_FLOWCOMPATOPS_H

#include <clip.hpp>

#include "evp/flow/flowsupportop.hpp"
#include "evp/flow/relaxflowopparams.hpp"

namespace evp {
using namespace clip;

class RelaxFlowOp {    
 private:
  RelaxFlowOpParams params_;
  NDArray<FlowSupportOpPtr,3> ops_;
  
 public:
  i32 iterations;
  f32 relaxStep;
  
  RelaxFlowOp(RelaxFlowOpParams &ps,
              i32 iters = 1,
              f32 relaxStep = 1.f)
  : params_(ps),
    ops_(ps.numOrientations, ps.numCurvatures, ps.numCurvatures),
    iterations(iters), relaxStep(relaxStep)
  {
    i32 nto = params_.numOrientations;
    i32 nks = params_.numCurvatures;
    f32 cs = params_.curvatureStep;
    
    for (i32 tii = 0; tii < nto; ++tii) {
      f64 ti = tii*params_.orientationStep;
      
      for (i32 ktii = 0; ktii < nks; ++ktii) {
        f64 kti = (ktii + 1 - params_.numCurvClasses)*cs;
        
        for (i32 knii = 0; knii < nks; ++knii) {
          f64 kni = (knii + 1 - params_.numCurvClasses)*cs;
          
          ops_(tii, ktii, knii) =
            FlowSupportOpPtr(params_.flowSupport(ti, kti, kni, params_));
        }
      }
    }
  }
  
  void apply(const NDArray<ImageBuffer,3>& input,
             NDArray<ImageBuffer,3>& output) {
    i32 nt = params_.numOrientations;
    i32 nk = params_.numCurvatures;
    
    output = NDArray<ImageBuffer,3>(nt, nk, nk);
  
    NDIndex<3> index;
    for (i32 iter = 0; iter < iterations; iter++) {
      const NDArray<ImageBuffer,3>& relaxSrc = iter == 0 ? input : output;
      NDArray<ImageBuffer,3> temp(nt, nk, nk);
      
      for (i32 tii = 0; tii < nt; tii++) {
        index[0] = tii;
        for (i32 ktii = 0; ktii < nk; ktii++) {
          index[1] = ktii;
          for (i32 knii = 0; knii < nk; knii++) {
            index[2] = knii;
            
            ImageBuffer support = ops_(tii%nt, ktii, knii)->apply(relaxSrc);
            MulAdd(relaxSrc[index], support, relaxStep, support);
            temp[index] = Bound(support, support);
          }
        }
      }
      
      output = temp;
    }
  }
};

}

#endif
