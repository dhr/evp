#pragma once
#ifndef EVP_CURVE_CURVESUPPORTOP_H
#define EVP_CURVE_CURVESUPPORTOP_H

#include <cmath>

#include <tr1/functional>
#include <vector>
#include <valarray>

#include <clip.hpp>

#include "evp/curve/curvetypes.hpp"
#include "evp/curve/curvecompatfuncs.hpp"
#include "evp/curve/curvecompatkern.hpp"
#include "evp/curve/relaxcurveop.hpp"
#include "evp/curve/loglin/llfuncs.hpp"
#include "evp/curve/loglin/llbufferops.hpp"
#include "evp/curve/loglin/llsupportops.hpp"
#include "evp/util/funcutils.hpp"
#include "evp/util/memutil.hpp"

namespace evp {
using namespace clip;

class CurveSupportOp {
  RelaxCurveOpParams params_;
  NDArray<ImageData,4> components_;
  NDArray<SparseImageData,4> sparseComponents_;
  NDArray<SparseImageBuffer,4> filterBufs_;
  bool preloadBuffers_;
  ContextID contextID_;
  
  void computePartPoints_(std::vector<f64> &ghist, i32 n,
                          std::vector<f64> &pnts) {
    pnts.resize(n - 1);
    std::valarray<f64> partSum(ghist.size());
    
    f64 sum = 0;
    for (i32 i = 0; i < i32(partSum.size()); ++i) {
      sum += ghist[i];
      partSum[i] = sum;
    }
    partSum /= sum;
    
    for (i32 i = 0; i < i32(pnts.size()); ++i) {
      f64 target = f64(i + 1)/n;
      i32 indx = 0;
      while (partSum[++indx] < target) continue;
      i32 trans = indx - i32(ghist.size())/2;
      pnts[i] = trans + 0.5 + (target - partSum[indx])/
      (partSum[indx] - partSum[indx - 1]);
    }
  }
  
  void balanceComponents_(f64 norm, ImageData &posSums, ImageData &negSums) {
    for (i32 kji = 0; kji < components_.size(3); ++kji) {
      for (i32 tji = 0; tji < components_.size(2); ++tji) {
        for (i32 ni = 0; ni < posSums.height(); ++ni) {
          for (i32 tani = 0; tani < posSums.width(); ++tani) {
            ImageData &kernel = components_(tani, ni, tji, kji);
            
            i32 kernElems = kernel.numElems();
            for (i32 i = 0; i < kernElems; i++) {
              kernel[i] *= norm/(kernel[i] >= 0 ?
                                 posSums(tani, ni) :
                                 negSums(tani, ni));
            }
          }
        }
      }
    }
  }
  
  void divideComponents_(NDArray<ImageData,3> &transDiffs,
                         std::vector<f64> &ghist,
                         RelaxCurveOpParams &p) {
    std::vector<f64> partPnts;
    computePartPoints_(ghist, p.numTangentialComponents, partPnts);
    StabilizedPartition partFunc(partPnts, 0, p.sigmaTransport,
                                 p.stabilizer, p.stabilizerDegree);
    
    ImageData posSums(p.numTangentialComponents, p.numNormalComponents);
    ImageData negSums(p.numTangentialComponents, p.numNormalComponents);
    
    std::vector<ImageData> parts(p.numTangentialComponents);
    std::vector<ImageData>::iterator it, end;
    for (it = parts.begin(), end = parts.end(); it != end; ++it)
      *it = ImageData(p.kernelSize, p.kernelSize);
    
    for (i32 ni = 0; ni < components_.size(1); ++ni) {
      for (i32 tji = 0; tji < components_.size(2); ++tji) {
        for (i32 kji = 0; kji < components_.size(3); ++kji) {
          ImageData &kernel = components_(0, ni, tji, kji);
          ImageData &trDiffData = transDiffs(ni, tji, kji);
          ImageData partSums(p.kernelSize, p.kernelSize);
          
          for (i32 el = 0; el < kernel.numElems(); ++el) {
            f64 gtrd = Gaussian(trDiffData[el], p.sigmaTransport, true);
            kernel[el] /= gtrd;
            
            for (i32 tani = 0;
                 tani < components_.size(0);
                 tani++) {
              partFunc.partIndx = tani;
              f64 part = partFunc(trDiffData[el]);
              partSums[el] += part/gtrd;
              parts[tani][el] = part;
            }
          }
          
          for (i32 tani = 1; tani < components_.size(0); tani++) {
            components_(tani, ni, tji, kji) = ImageData(kernel, true);
          }
          
          for (i32 tani = 0; tani < components_.size(0); tani++) {
            ImageData &component = components_(tani, ni, tji, kji);
            
            for (i32 el = 0; el < kernel.numElems(); el++) {
              if (partSums[el] > 0) {
                if (parts[tani][el] != 0) {
                  component[el] *=
                  parts[tani][el]/partSums[el];
                  if (component[el] > 0) {
                    posSums(tani, ni) += component[el];
                  }
                  else {
                    negSums(tani, ni) -= component[el];
                  }
                }
                else {
                  component[el] = 0;
                }
              }
            }
          }
        }
      }
    }
    
    balanceComponents_(1.f/p.numTangentialComponents, posSums, negSums);
  }
  
  void initSparseComponents() {
    for (i32 i = 0; i < components_.numElems(); i++)
      sparseComponents_[i] = SparseImageData(components_[i]);
  }
  
 public:
  CurveSupportOp(f64 ti, f64 ki, RelaxCurveOpParams &p)
  : params_(p),
    components_(p.numTangentialComponents, p.numNormalComponents,
                p.numTotalOrientations, p.numCurvatures),    
    sparseComponents_(p.numTangentialComponents, p.numNormalComponents,
                      p.numTotalOrientations, p.numCurvatures),
    filterBufs_(p.numTangentialComponents, p.numNormalComponents,
                p.numTotalOrientations, p.numCurvatures),
    preloadBuffers_(true), contextID_(-1)
  {
    i32 halfSize = p.kernelSize/2;
    std::tr1::array<f64, 4> i = {{halfSize, halfSize, ti, ki}};
    std::tr1::array<f64, 4> j;
    
    i32 nto = p.numTotalOrientations;
    i32 nk = p.numCurvatures;
    i32 nnc = p.numNormalComponents;
    
    std::vector<f64> ghist(p.kernelSize);
    ImageData posSums(1, p.numNormalComponents);
    ImageData negSums(1, p.numNormalComponents);
    
    NDArray<ImageData,3> transDiffs(p.numNormalComponents,
                                    nto, p.numCurvatures);
    
    for (i32 tji = 0; tji < nto; ++tji) {
      j[2] = tji*p.orientationStep;
      
      for (i32 kji = 0; kji < nk; ++kji) {
        j[3] = (kji + 1 - i32(p.numCurvClasses))*p.curvatureStep;
        
        for (i32 ni = 0; ni < nnc; ++ni) {
          CurveCompatKern kernel =
            CurveCompatKern(i, j, ni, p, transDiffs(ni, tji, kji), ghist);
          components_(0, ni, tji, kji) = kernel;
          
          for (i32 el = 0; el < kernel.numElems(); ++el) {
            if (kernel[el] > 0)
              posSums(0, ni) += kernel[el];
            else if (kernel[el] < 0)
              negSums(0, ni) -= kernel[el];
          }
        }
      }
    }
    
    balanceComponents_(1, posSums, negSums); 
    divideComponents_(transDiffs, ghist, p);
    initSparseComponents();
  }
  
  const NDArray<ImageData,4>& components() const {
    return components_;
  }
  
  ImageBuffer apply(const CurveBuffers& inputs, ImageBuffer output) {
    using namespace std::tr1;
    using namespace std::tr1::placeholders;
    
    i32 nto = params_.numTotalOrientations;
    i32 ncs = params_.numCurvatures;
    i32 nnc = params_.numNormalComponents;
    i32 ntc = params_.numTangentialComponents;
  
    ImBufList stack;
    PopAdaptor popper(stack);
    PushAdaptor pusher(stack);
    
    if (preloadBuffers_ && contextID_ != CurrentContextID()) {
      LoadSparseFilters(sparseComponents_.begin(),
                        sparseComponents_.end(),
                        filterBufs_.begin());
      contextID_ = CurrentContextID();
    }
    
    NDIndex<2> srcIndx;
    for (i32 tani = 0; tani < ntc; ++tani) {
      for (i32 ni = 0; ni < nnc; ++ni) {
        for (i32 tji = 0; tji < nto; ++tji) {
          srcIndx[0] = tji;
          
          for (i32 kji = 0; kji < ncs; ++kji) {
            srcIndx[1] = kji;
            
            if (preloadBuffers_) {
              const SparseImageBuffer& filt =
                filterBufs_(tani, ni, tji, kji);
              pusher.output(Filter(inputs[srcIndx], filt));
            }
            else {
              const SparseImageData& filt =
                sparseComponents_(tani, ni, tji, kji);
              pusher.output(Filter(inputs[srcIndx], filt));
            }
          }
        }
        
        pusher.output(Merge(Add, nto*ncs, popper));
      }
      
      pusher.output(LLMerge(LLAnd, nnc, params_.logLinDegree,
                            params_.adapt, 1.f, popper));
    }
    
    CircularAdaptor circ(stack);
    ImageBuffer sum = Merge(Add, ntc, circ);
    circ.reset();
    Map(bind(Stabilize, _1, sum, ntc, params_.stabilizer, _2), ntc, circ);
    
    i32 halfntc = ntc/2;
    
    if (ntc != 2) {
      // To support more tangential components, the surround operation would
      // have to be extended to support more than two arguments.
      assert(ntc == 4 && "Only two or four tangential components supported");
      
      ImageBuffer tcs[4] = {popper.next(), popper.next(),
                            popper.next(), popper.next()};
      
      pusher.output(Surround(tcs[1], tcs[0], params_.logLinDegree));
      pusher.output(Surround(tcs[2], tcs[3], params_.logLinDegree));
    }
    
    return LLMerge(LLAnd, ntc - 2*halfntc + 2,
                   params_.logLinDegree, params_.adapt, 1.f,
                   popper, output);
  }
  
  ImageBuffer apply(NDArray<ImageBuffer,2>& input) {
    return apply(input, ~input[0]);
  }
};

typedef std::tr1::shared_ptr<CurveSupportOp> CurveSupportOpPtr;

}

#endif
