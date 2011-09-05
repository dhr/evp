#ifndef EVP_CURVE_EDGELINEOP_H
#define EVP_CURVE_EDGELINEOP_H

#include <cmath>

#include <clip.hpp>

#include "evp/curve/curvecompatfuncs.hpp"
#include "evp/curve/suppresslineedgesopparams.hpp"

namespace evp {
using namespace clip;

class LineEdgeLocatorOp : public Monitorable {
  SuppressLineEdgesOpParams params_;
  NDArray<ImageData,1> kernels_;
  NDArray<SparseImageData,1> sparseKernels_;
  NDArray<SparseImageBuffer,1> sparseBufs_;
  bool preloadBuffers_;
  ContextID contextID_;
  
  ImageData makeLineEdgeLocatorKern(f64 ti, f64 tj, f64 thresh,
                                    SuppressLineEdgesOpParams& p) {
    std::tr1::array<f64,4> i, j;
    i[0] = p.kernelSize/2; i[1] = p.kernelSize/2; i[2] = ti; i[3] = 0;
    j[2] = tj; j[3] = 0;
    
    ImageData kernel(p.kernelSize, p.kernelSize);
    
    ProjDiffs diffs;
    
    i32 offSgn = cmod(tj) >= 0 ? 1 : -1;
    f64 xoff = -offSgn*sin(ti), yoff = offSgn*cos(ti);
    for (i32 yji = 0; yji < p.kernelSize; ++yji) {
      j[1] = yji + yoff;
      for (i32 xji = 0; xji < p.kernelSize; ++xji, ++j[0]) {
        j[0] = xji + xoff;
        
        ProjectionDifference(i, j, 1, diffs);
        
        f64 gxy = Gaussian(diffs.xy, p.sigmaXY, false);
        f64 gth = Gaussian(diffs.theta, p.sigmaTheta, false);
        f64 gtr = Gaussian(diffs.transport, p.sigmaTransport, false);
        f64 gprod = gxy*gth*gtr;
        kernel(xji, yji) = gprod >= thresh ? gprod : 0;
      }
    }
    
    return kernel;
  }

 public:
  LineEdgeLocatorOp(f64 ti, SuppressLineEdgesOpParams& p)
  : params_(p),
    kernels_(p.numTotalOrientations),
    sparseKernels_(p.numTotalOrientations),
    sparseBufs_(p.numTotalOrientations),
    preloadBuffers_(true),
    contextID_(-1)
  {
//    f64 sum = 0;
    
    for (i32 tji = 0; tji < p.numTotalOrientations; ++tji) {
      f64 tj = tji*p.orientationStep;
      ImageData kern = makeLineEdgeLocatorKern(ti, tj, 0.05, p);
//      sum += kern.data().sum();
      kernels_[tji] = kern;
    }
    
    for (i32 tji = 0; tji < p.numTotalOrientations; ++tji) {
//      kernels_[tji].data() /= sum;
      sparseKernels_[tji] = SparseImageData(kernels_[tji]);
    }
  }
  
  ImageBuffer apply(const CurveBuffers& lines, ImageBuffer output) {
    i32 nts = lines.size(0);
    i32 nks = lines.size(1);
    
    ImageBuffer amounts;
    
    ImBufList bufStack;
    PopAdaptor popper(bufStack);
    PushAdaptor pusher(bufStack);
    
    if (preloadBuffers_ && contextID_ != CurrentContextID()) {
      LoadSparseFilters(sparseKernels_.begin(),
                        sparseKernels_.end(),
                        sparseBufs_.begin());
      contextID_ = CurrentContextID();
    }
    
    for (i32 tji = 0; tji < nts; ++tji) {
      for (i32 kji = 0; kji < nks; ++kji)
        pusher.output(lines(tji, kji));
      
      ImageBuffer lineLocs = Merge(Max, nks, popper);
      pusher.output(Filter(lineLocs, sparseBufs_[tji]));
    }
    
    return Merge(Add, nts, popper, output);
  }
  
  ImageBuffer apply(const CurveBuffers& lines) {
    return apply(lines, ~lines[0]);
  }
};

typedef std::tr1::shared_ptr<LineEdgeLocatorOp> LineEdgeLocatorOpPtr;

}

#endif
