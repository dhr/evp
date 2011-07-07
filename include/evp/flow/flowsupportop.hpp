#pragma once
#ifndef EVP_FLOW_FLOWCONNECTIONSOP_H
#define EVP_FLOW_FLOWCONNECTIONSOP_H

#include <clip.hpp>

#include "evp/util/mathutil.hpp"
#include "evp/flow/flowmodel.hpp"
#include "evp/flow/relaxflowopparams.hpp"

namespace evp {
using namespace clip;

class FlowSupportOp {
  friend FlowSupportOp *
  CreateInhibitionlessFlowSupportOp(f64, f64, f64, RelaxFlowOpParams &);
  
 protected:
  RelaxFlowOpParams &params_;
  FlowModel model_;
  NDArray<ImageData,3> kernels_;
  NDArray<SparseImageBuffer,3> kernBufs_;
  f32 ti_;
  f32 kti_;
  f32 kni_;
  
  virtual void calculateConnections() {
    f32 maxExcitation = 0.f;
    
    i32 nto = params_.numOrientations;
    i32 nk = params_.numCurvatures;
    i32 kernSize = params_.kernelSize;
    i32 radius = kernSize/2;
    i32 radSquared = radius*radius;
    
    f32 dt = params_.orientationStep;
    f32 dk = params_.curvatureStep;
    
    i32 nsamps = params_.subsamples;
    
    f64 sBndry = f64(nsamps/2)/nsamps;
    f64 sIncr = 1./nsamps;
    f64 stBndry = sBndry*dt;
    f64 stIncr = sIncr*dt;
    f64 skBndry = sBndry*dk;
    f64 skIncr = sIncr*dk;
    
    // The following five for-loops implement subsampling in each dimension
    for (f64 sx = -sBndry; sx <= sBndry + sIncr/2; sx += sIncr) {
    for (f64 sy = -sBndry; sy <= sBndry + sIncr/2; sy += sIncr) {
    for (f64 st = -stBndry; st <= stBndry + stIncr/2; st += stIncr) {
    for (f64 skt = -skBndry; skt <= skBndry + skIncr/2; skt += skIncr) {
    for (f64 skn = -skBndry; skn <= skBndry + skIncr/2; skn += skIncr) {
      model_.set(sx, sy, ti_ + st, kti_ + skt, kni_ + skn);
              
      for (i32 yi = 0; yi < kernSize; yi++) {
        i32 yoff = yi - radius;
        f64 y = yoff + sy;
        
        for (i32 xi = 0; xi < kernSize; xi++) {
          i32 xoff = xi - radius;
          
          if (xoff*xoff + yoff*yoff > radSquared) {
            continue;
          }
          
          f64 x = xoff + sx;
          f64 theta, kt, kn;
          
          if (!model_.valuesAt(x, y, &theta, &kt, &kn)) {
            continue;
          }
          
          i32 tji = i32(round(theta/dt));
          
          if (params_.numPis == 1) {
            if (tji < 0) {
              tji += nto;
              kt = -kt;
              kn = -kn;
            }
            else if (tji >= nto) {
              tji -= nto;
              kt = -kt;
              kn = -kn;
            }
          }
          
          tji = clamp(tji, 0, nto - 1);
          
          i32 ktji = i32(round(kt/dk));
          ktji = clamp(ktji + nk/2, 0, nk - 1);
          
          i32 knji = i32(round(kn/dk));
          knji = clamp(knji + nk/2, 0, nk - 1);
          
          f32& kernVal = kernels_(tji, ktji, knji)(xi, yi);
          
          kernVal += 1;
          maxExcitation += 1;
        }
      }
    }}}}}
    
    for (i32 i = 0; i < kernels_.numElems(); ++i) {
      kernels_[i].data() /= maxExcitation;
    }
  }
  
  FlowSupportOp(f64 ti, f64 kti, f64 kni, RelaxFlowOpParams &params)
  : params_(params),
    kernels_(params.numOrientations,
             params.numCurvatures,
             params.numCurvatures),
    kernBufs_(params.numOrientations,
              params.numCurvatures,
              params.numCurvatures),
    ti_(ti),
    kti_(kti),
    kni_(kni)
  {
    for (i32 i = 0; i < kernels_.numElems(); i++)
      kernels_[i] = ImageData(params.kernelSize, params.kernelSize);
  }
  
 public:
  virtual ~FlowSupportOp() {}
  
  const NDArray<ImageData,3> &kernels() const {
    return kernels_;
  }
  
  ImageBuffer apply(const NDArray<ImageBuffer,3>& input, ImageBuffer output) {
    ImBufList stack;
    PopAdaptor popper(stack);
    PushAdaptor pusher(stack);
    
    if (!kernBufs_[0].valid())
      LoadSparseFilters(kernels_.begin(), kernels_.end(), kernBufs_.begin());
  
    i32 nto = params_.numOrientations;
    i32 ncs = params_.numCurvatures;
    
    NDIndex<3> srcIndx;
    for (i32 tji = 0; tji < nto; ++tji) {
      srcIndx[0] = tji;
      
      for (i32 ktji = 0; ktji < ncs; ++ktji) {
        srcIndx[1] = ktji;
        
        for (i32 knji = 0; knji < ncs; ++knji) {
          srcIndx[2] = knji;
          
          pusher.output(Filter(input[srcIndx], kernBufs_[srcIndx]));
        }
      }
      
      pusher.output(Merge(Add, ncs*ncs, popper));
    }
    
    Merge(Add, nto, popper, output);
        
    if (params_.minSupport != 0 || params_.maxSupport != 1) {
      Rescale(output, params_.minSupport, params_.maxSupport,
              0, 1, false, output);
    }
    
    return output;
  }
  
  ImageBuffer apply(const NDArray<ImageBuffer,3>& input) {
    return apply(input, ~input[0]);
  }
};

typedef std::tr1::shared_ptr<FlowSupportOp> FlowSupportOpPtr;

FlowSupportOp *
CreateInhibitionlessFlowSupportOp(f64 ti, f64 kti, f64 kni,
                               RelaxFlowOpParams &params) {
  FlowSupportOp *op = new FlowSupportOp(ti, kti, kni, params);
  op->calculateConnections();
  return op;
}

}

#endif
