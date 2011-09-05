#ifndef EVP_CURVE_SUPPRESSLINEEDGESOP_H
#define EVP_CURVE_SUPPRESSLINEEDGESOP_H

#include "evp/curve/relaxcurveopparams.hpp"

namespace evp {
using namespace clip;

class SuppressLineEdgesOp : public Monitorable {
  NDArray<LineEdgeLocatorOpPtr,1> locatorOps_;
  f32 delta_;
  
 public:
  SuppressLineEdgesOp(SuppressLineEdgesOpParams& p, f32 delta = 2.f)
  : locatorOps_(p.numTotalOrientations), delta_(delta)
  {
    for (i32 tii = 0; tii < p.numTotalOrientations; ++tii) {
      locatorOps_[tii] = LineEdgeLocatorOpPtr
        (new LineEdgeLocatorOp(tii*p.orientationStep, p));
    }
  }
  
  CurveBuffersPtr apply(const CurveBuffers& edges, const CurveBuffers& lines) {
    i32 nts = edges.size(0);
    i32 nks = edges.size(1);
    
    CurveBuffersPtr outputPtr(new CurveBuffers(nts, nks));
    
    for (i32 tii = 0; tii < nts; ++tii) {
      ImageBuffer suppression = locatorOps_[tii]->apply(lines);
      Threshold(suppression, 0.01, suppression);
      for (i32 kii = 0; kii < nks; ++kii) {
        (*outputPtr)(tii, kii) = MulAdd(edges(tii, kii), suppression, -delta_);
      }
    }
    
    return outputPtr;
  }
};

}

#endif
