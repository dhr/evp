#pragma once
#ifndef EVP_CURVE_LOGLIN_LLINITOPS_H
#define EVP_CURVE_LOGLIN_LLINITOPS_H

#include <sstream>

#include <clip.hpp>

#include "evp/curve/curvetypes.hpp"
#include "evp/curve/curvefeaturetypes.hpp"
#include "evp/curve/loglin/llbasissets.hpp"
#include "evp/curve/loglin/llendstoppedop.hpp"
#include "evp/curve/loglin/llinitopparams.hpp"
#include "evp/curve/loglin/llsimplecellop.hpp"
#include "evp/util/mathutil.hpp"
#include "evp/util/memutil.hpp"
#include "evp/util/monitorable.hpp"

namespace evp {
using namespace clip;

class LLInitOps : public Monitorable {
 private:
  const LLInitOpParams params_;
  NDArray<LLCellOpPtr,2> ops_;
  
 public:
  LLInitOps(LLInitOpParams &ps)
  : params_(ps), ops_(ps.numOrientations, ps.numCurvClasses) {
    LLBasisSetPtr excNormalBasisSet;
    LLBasisSetPtr posKInhNormalBasisSet;
    LLBasisSetPtr negKInhNormalBasisSet;
    LLBasisSetPtr excTangentialBasisSet;
    LLBasisSetPtr inhTangentialBasisSet;
    
    if (params_.featureType == Edges) {
      excNormalBasisSet =
        LLBasisSetPtr(new LLEdgeBasisSet(params_.normalSigma,
                                         params_.normalOffset));
      
      if (params_.numCurvClasses > 1) {
        posKInhNormalBasisSet =
          LLBasisSetPtr(new LLInhEdgeBasisSet(params_.normalSigma,
                                              params_.normalOffset, 1));
        negKInhNormalBasisSet =
          LLBasisSetPtr(new LLInhEdgeBasisSet(params_.normalSigma,
                                              params_.normalOffset, -1));
      }
    }
    else {
      excNormalBasisSet =
        LLBasisSetPtr(new LLLineBasisSet(params_.normalSigma,
                                         params_.normalOffset));
      
      if (ps.numCurvClasses > 1) {
        posKInhNormalBasisSet =
          LLBasisSetPtr(new LLInhLineBasisSet(params_.normalSigma,
                                              params_.normalOffset, 1));
        negKInhNormalBasisSet =
          LLBasisSetPtr(new LLInhLineBasisSet(params_.normalSigma,
                                              params_.normalOffset, -1));
      }
    }
    
    for (i32 k = 0; k < params_.numCurvClasses; ++k) {
      const LLInitOpParams::CurvatureParams &cp = params_.curvatureParams[k];
      excTangentialBasisSet =
        LLBasisSetPtr(new LLStabilizerBasisSet(cp.excComponents,
                                               cp.excTangentialSigma,
                                               params_.excStabilizer,
                                               params_.stabilizerDegree));
      
      if (k > 0) {
        inhTangentialBasisSet =
          LLBasisSetPtr(new LLStabilizerBasisSet(cp.inhComponents,
                                                 cp.inhTangentialSigma,
                                                 params_.inhStabilizer,
                                                 params_.stabilizerDegree));
      }
      
      LLSimpleCellOpPtr excComponent;
      LLSimpleCellOpPtr posKInhComponent;
      LLSimpleCellOpPtr negKInhComponent;
      for (i32 t = 0; t < params_.numOrientations; ++t) {
        f32 orientation = t*params_.orientationStep;
        
        excComponent =
          LLSimpleCellOpPtr(new LLSimpleCellOp(orientation,
                                               excNormalBasisSet,
                                               excTangentialBasisSet,
                                               cp.excScaling,
                                               params_.logLinDegree,
                                               params_.adapt));
        
        if (k > 0) {
          posKInhComponent =
            LLSimpleCellOpPtr(new LLSimpleCellOp(orientation,
                                                 posKInhNormalBasisSet,
                                                 inhTangentialBasisSet,
                                                 cp.inhScaling,
                                                 params_.logLinDegree,
                                                 params_.adapt));
          
          negKInhComponent =
            LLSimpleCellOpPtr(new LLSimpleCellOp(orientation,
                                                 negKInhNormalBasisSet,
                                                 inhTangentialBasisSet,
                                                 cp.inhScaling,
                                                 params_.logLinDegree,
                                                 params_.adapt));
          
          ops_(t, k) = LLCellOpPtr(new LLEndstoppedOp(excComponent,
                                                      posKInhComponent,
                                                      negKInhComponent));
        }
        else
          ops_(t, k) = excComponent;
      }
    }
  }
  
  CurveBuffersPtr apply(const ImageBuffer& image) {
    i32 nt = params_.numOrientations;
    i32 nk = params_.numCurvatures;
    i32 nkc = params_.numCurvClasses;
    
    CurveBuffersPtr outputPtr(new CurveBuffers(2*nt, nk));
    CurveBuffers& output = *outputPtr;
    
    for (i32 t = 0; t < nt; ++t) {
      for (i32 k = 0; k < nkc; ++k) {
        ImageBuffer responses[4];
        ops_(t, k)->apply(image, responses);
        
        for (i32 i = 0; i < 2; ++i) {
          i32 o = i*nt;
          i32 s = i == 0 ? 1 : -1;
          
          output(t + o, nkc - 1 + s*k) = Bound(responses[i]);
          
          if (k > 0)
            output(t + o, nkc - 1 - s*k) = Bound(responses[i + 2]);
        }
        
        setProgress(f32(t*nkc + k + 1)/(nt*nkc));
      }
    }
    
    return outputPtr;
  }
};

}

#endif
