#pragma once
#ifndef EVP_FLOW_UNIFORMINHFLOWOP_H
#define EVP_FLOW_UNIFORMINHFLOWOP_H

namespace evp {
using namespace clip;

class UniformInhibitionFlowSupportOp : public FlowSupportOp {
  friend FlowSupportOp *
  CreateUniformInhibitionFlowSupportOp(f64, f64, f64,
                                       RelaxFlowOpParams &);
  
 protected:
  void calculateConnections() {
    f32 totalExcitation = calculateExcitatoryConnections();
    f32 totalInhibition = calculateInhibitoryConnections();
    balanceConnections(totalExcitation, totalInhibition/params_.inhRatio);
  }
  
  f32 calculateExcitatoryConnections() {
    FlowSupportOp::calculateConnections();
    
    f32 totalExcitation = 0.f;
    
    for (i32 i = 0; i < kernels_.numElems(); ++i) {
      totalExcitation += kernels_[i].data().sum();
    }
    
    return totalExcitation;
  }
  
  f32 calculateInhibitoryConnections() {
    i32 nto = params_.numTotalOrientations;
    i32 nk = params_.numCurvatures;
    i32 kernSize = params_.kernelSize;
    i32 radius = kernSize/2;
    i32 radSquared = radius*radius;
    
    f64 dt = params_.orientationStep;
    
    f64 falloffSigma = M_PI/8.;
    
    f32 totalInhibition = 0.f;
    
    for (i32 tji = 0; tji < nto; tji++) {
      for (i32 ktji = 0; ktji < nk; ktji++) {
        for (i32 knji = 0; knji < nk; knji++) {
          ImageData &k = kernels_(tji, ktji, knji);
          
          f64 inh;
          f64 tdiff2 = cmod(tji*dt - ti_ + M_PI_2, M_PI);
          tdiff2 = cmod(tji*dt - ti_, M_PI);
          tdiff2 *= tdiff2;
          inh = -exp(-tdiff2/(2*falloffSigma));
//          inh = -1;
          
          i32 i = 0;
          for (i32 yi = 0; yi < kernSize; ++yi) {
            f64 yoff = yi - radius;
            
            for (i32 xi = 0; xi < kernSize; ++xi, ++i) {
              f64 xoff = xi - radius;
              
              if (xoff*xoff + yoff*yoff > radSquared)
                continue;
              
              if (k[i] == 0) {
                k[i] = inh;
                totalInhibition -= inh; // since inh < 0
              }
            }
          }
        }
      }
    }
    
    return totalInhibition;
  }
  
  void balanceConnections(f32 totalExcitation, f32 totalInhibition) {
    for (i32 i = 0; i < kernels_.numElems(); ++i) {
      ImageDataValues &data = kernels_[i].data();
      
      for (i32 j = 0; j < i32(data.size()); ++j) {
        data[j] /= data[j] >= 0 ? totalExcitation : totalInhibition;
      }
    }
  }
  
 public:
  UniformInhibitionFlowSupportOp(f64 ti, f64 kti, f64 kni,
                                     RelaxFlowOpParams &params_)
  : FlowSupportOp(ti, kti, kni, params_) {}
};

FlowSupportOp *
CreateUniformInhibitionFlowSupportOp(f64 ti, f64 kti, f64 kni,
                                     RelaxFlowOpParams &params) {
  UniformInhibitionFlowSupportOp * op =
    new UniformInhibitionFlowSupportOp(ti, kti, kni, params);
  op->calculateConnections();
  return op;
}

}

#endif
