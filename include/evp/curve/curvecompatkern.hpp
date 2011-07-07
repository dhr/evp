#pragma once
#ifndef EVP_CURVE_CURVECOMPATKERN_H
#define EVP_CURVE_CURVECOMPATKERN_H

#include <cmath>

#include <tr1/array>

#include "evp/curve/relaxcurveopparams.hpp"
#include "evp/util/funcutils.hpp"

namespace evp {
using namespace clip;

class CurveCompatKern : public ImageData {
 public:
  CurveCompatKern(std::tr1::array<f64, 4> &i,
                  std::tr1::array<f64, 4> &j,
                  i32 ni, RelaxCurveOpParams &p,
                  ImageData &trDiffData, std::vector<f64> &ghist)
  : ImageData(p.kernelSize, p.kernelSize)
  {
    f64 offsetSign = ni%2 == 0 ? -1 : 1;
    f64 offset = offsetSign*p.offset;
    
    f64 renormFactor = Gaussian(-1, 1, true);
    
    i32 halfSize = p.kernelSize/2;
    
    ProjDiffs diffs;
    
    trDiffData = ImageData(p.kernelSize, p.kernelSize);
    
    j[1] = 0;
    for (i32 yji = 0; yji < p.kernelSize; yji++, j[1] += 1) {
      j[0] = 0;
      for (i32 xji = 0; j[0] < p.kernelSize; xji++, j[0] += 1) {
        ProjectionDifference(i, j, p.numTotalOrientations, diffs);
        trDiffData(xji, yji) = diffs.transport;
        
        f64 expansion = 1 + p.dilation*fabs(diffs.transport);
        f64 xSigmaXY = expansion*p.sigmaXY;
        f64 xSigmaTheta = expansion*p.orientationStep*p.sigmaTheta;
        f64 xSigmaKappa = expansion*p.curvatureStep*p.sigmaKappa;
        
        f64 gxy = Gaussian(diffs.xy, xSigmaXY, false);
        f64 gth = Gaussian(diffs.theta, xSigmaTheta, false);
        f64 gk = Gaussian(diffs.kappa, xSigmaKappa, false);
        f64 gtr = Gaussian(diffs.transport, p.sigmaTransport, false);
        
        f64 gprod = gxy*gth*gk*gtr;
        
        bool selected = diffs.transport > -halfSize - 0.5 &&
        diffs.transport < halfSize + 0.5;
        if (selected && ni == 0) {
          i32 histIndex = round(diffs.transport) + halfSize;
          ghist[histIndex] += gprod;
        }
        
        if (ni < 2) {
          (*data_)[yji*width_ + xji] = (f32)
          (offsetSign*gth*gk*gtr*
           DGaussian(diffs.xy/xSigmaXY - expansion*offset, 1, 1)/renormFactor);
        }
        else if (ni < 4) {
          (*data_)[yji*width_ + xji] = (f32)
          (offsetSign*gxy*gk*gtr
           *DGaussian(diffs.theta/xSigmaTheta - expansion*offset, 1, 1)
           /renormFactor);
        }
        else {
          (*data_)[yji*width_ + xji] = (f32)
          (offsetSign*gxy*gth*gtr
           *DGaussian(diffs.kappa/xSigmaKappa - expansion*offset, 1, 1)
           /renormFactor);
        }
      }
    }
  }
};

}

#endif
