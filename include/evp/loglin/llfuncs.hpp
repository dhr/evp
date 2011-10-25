#pragma once
#ifndef EVP_LOGLIN_LLFUNCS_H
#define EVP_LOGLIN_LLFUNCS_H

#include <cmath>

#include <vector>

#include "evp/util/gaussian.hpp"
#include "evp/loglin/llbasis.hpp"

namespace evp {
using namespace clip;

class NthDOfG : public SimpleFunc {
 public:
  f64 sigma;
  i32 n;
  
  NthDOfG(f64 s, i32 nth) :
  sigma(s), n(nth) {}
  
  f64 operator()(f64 x) { return DGaussian(x, sigma, n); }
};

inline f64 SmoothPart(f64 x, f64 degree) {
  x *= degree;
  
  if (x < -0.5) return 0;
  if (x > 0.5) return 1;
  
  f64 temp = exp(-1/(0.5 + x));
  return temp/(temp + exp(-1/(0.5 -x )));
}

class StabilizedPartition : public SimpleFunc {
 public:
  std::vector<f64> partPoints;
  i32 partIndx;
  f64 sigma;
  f64 degree;
  f64 stabilizer;
  
  StabilizedPartition(std::vector<f64> pnts, i32 i,
                      f64 s, f64 stab, f64 d)
  : partPoints(pnts), partIndx(i), sigma(s), degree(d), stabilizer(stab) {}
  
  f64 operator()(f64 x) {
    f64 r = Gaussian(x, sigma);
    
    f64 sd = sigma*degree;
    if (partIndx == 0)
      r *= SmoothPart(partPoints[partIndx] - x, 1/sd);
    else if (partIndx == i32(partPoints.size()))
      r *= SmoothPart(x - partPoints[partIndx - 1], 1/sd);
    else {
      f64 left = SmoothPart(x - partPoints[partIndx - 1], 1/sd);
      f64 right = SmoothPart(partPoints[partIndx] - x, 1/sd);
      r *= left + right - 1;
    }
    
    if (stabilizer > 0) {
      if (partIndx == (i32(partPoints.size()) + 1)/2 - 1)
        r += stabilizer*sigma*DGaussian(x, sigma, 1);
      else if (partIndx == (i32(partPoints.size()) + 1)/2)
        r -= stabilizer*sigma*DGaussian(x, sigma, 1);
    }
    
    return r;
  }
};

}

#endif
