#pragma once
#ifndef EVP_FLOW_FLOWCOMPATOPPARAMS_H
#define EVP_FLOW_FLOWCOMPATOPPARAMS_H

namespace evp {
using namespace clip;

class FlowSupportOp;
struct RelaxFlowOpParams;

FlowSupportOp *
CreateInhibitionlessFlowSupportOp(f64, f64, f64, RelaxFlowOpParams &);

FlowSupportOp *
CreateUniformInhibitionFlowSupportOp(f64, f64, f64,
                                     RelaxFlowOpParams &);

struct RelaxFlowOpParams {
  i32 numOrientations;
  i32 numCurvatures;
  f32 curvatureStep;
  i32 kernelSize;
  i32 subsamples;
  
  FlowSupportOp *
    (*flowSupport)(f64, f64, f64, RelaxFlowOpParams &);
  
  // Inferred parameters
  
  f32 orientationStep;
  i32 numTotalOrientations;
  i32 numPis;
  i32 numCurvClasses;
  f32 minSupport;
  f32 maxSupport;
  f32 inhRatio;
  
  RelaxFlowOpParams(i32 nt = 8, i32 nk = 5)
  : numOrientations(nt),
    numCurvatures(nk),
    curvatureStep(0.1),
    kernelSize(9),
    subsamples(3),
    flowSupport(CreateUniformInhibitionFlowSupportOp),
//    flowSupport(CreateInhibitionlessFlowSupportOp),
    orientationStep(M_PI/nt),
    numPis(1),
    numCurvClasses(nk/2 + 1),
    minSupport(0.1), 
    maxSupport(1),
    inhRatio(5) {}
};

}

#endif
