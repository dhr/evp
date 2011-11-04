#pragma once
#ifndef EVP_LOGLIN_LLBASISSETS_H
#define EVP_LOGLIN_LLBASISSETS_H

#include <cmath>

#include <valarray>

#include "evp/loglin/llbasisset.hpp"
#include "evp/loglin/llfuncs.hpp"
#include "evp/loglin/llbufferops.hpp"
#include "evp/loglin/llsupportops.hpp"

namespace evp {
using namespace clip;

class LLEdgeBasisSet : public LLBasisSet {
 protected:
  using LLBasisSet::bases_;
  using LLBasisSet::kernSize_;
  
 public:
  LLEdgeBasisSet(f64 sigma, f64 offset)
  : LLBasisSet(std::vector<LLBasis>(5), i32(floor(6*sigma + 2*offset + 1)))
  {
    bases_[0] = LLBasis(new NthDOfG(sigma, 1), kernSize_, -1, 0);
    bases_[1] = LLBasis(new NthDOfG(sigma, 2), kernSize_, 1, -offset);
    bases_[2] = LLBasis(new NthDOfG(sigma, 2), kernSize_, -1, offset);
    bases_[3] = LLBasis(new NthDOfG(sigma, 4), i32(kernSize_/sqrt(2.f)),
                        -1, -offset);
    bases_[4] = LLBasis(new NthDOfG(sigma, 4), i32(kernSize_/sqrt(2.f)),
                        1, offset);
    normalize();
  }
  
  void combine(f64 deg, bool adapt, InputAdaptor& i, OutputAdaptor& o) {
    o.output(LLMerge(LLAnd, 5, deg, adapt, 1.f/5, i));
  }
};

class LLInhEdgeBasisSet : public LLBasisSet {
 public:
  LLInhEdgeBasisSet(f64 sigma, f64 offset, i32 curvSign)
  : LLBasisSet(std::vector<LLBasis>(1),	i32(floor(6*sigma + 2*offset + 1)))
  {
    bases_[0] = LLBasis(new NthDOfG(sigma, 2), kernSize_,
                        curvSign > 0 ? -1 : 1,
                        curvSign > 0 ? offset : -offset);
    normalize();
  }
  
  void combine(f64, bool, InputAdaptor&, OutputAdaptor&) {
  }
};

class LLLineBasisSet : public LLBasisSet {
 public:
  LLLineBasisSet(f64 sigma, f64 offset)
  : LLBasisSet(std::vector<LLBasis>(4),	i32(floor(6*sigma + 2*offset + 1)))
  {
    bases_[0] = LLBasis(new NthDOfG(sigma, 1), kernSize_, 1, -offset);
    bases_[1] = LLBasis(new NthDOfG(sigma, 1), kernSize_, -1, offset);
    bases_[2] = LLBasis(new NthDOfG(sigma/sqrt(2.f), 3),
                        kernSize_, -1, -offset);
    bases_[3] = LLBasis(new NthDOfG(sigma/sqrt(2.f), 3),
                        kernSize_, 1, offset);
    normalize();
  }
  
  void combine(f64 deg, bool adapt, InputAdaptor& i, OutputAdaptor& o) {
    o.output(LLMerge(LLAnd, 4, deg, adapt, 1.f/4, i));
  }
};

class LLInhLineBasisSet : public LLBasisSet {
 public:
  LLInhLineBasisSet(f64 sigma, f64 offset, i32 curvSign)
  : LLBasisSet(std::vector<LLBasis>(1),	i32(floor(6*sigma + 2*offset + 1)))
  {
    bases_[0] = LLBasis(new NthDOfG(sigma, 1), kernSize_,
                        curvSign > 0 ? -1 : 1,
                        curvSign > 0 ? offset : -offset);
    normalize();
  }
  
  void combine(f64, bool, InputAdaptor&, OutputAdaptor&) {
  }
};

class LLStabilizerBasisSet : public LLBasisSet {
 public:
  LLStabilizerBasisSet() {}
  
  LLStabilizerBasisSet(i32 n, f64 sigma, f64 stabilizer, f64 degree)
  : LLBasisSet(std::vector<LLBasis>(n),	i32(floor(6*sigma + 1)))
  {
    std::vector<f64> pts(n - 1);
    computePartPoints(sigma, pts);
    for (i32 i = 0; i < n; ++i) {
      SimpleFunc *f =
        new StabilizedPartition(pts, i, sigma, stabilizer, degree);
      bases_[i] = LLBasis(f, kernSize(), 1, 0, false);
    }
  }
  
  void combine(f64 deg, bool adapt, InputAdaptor& i, OutputAdaptor& o) {
    o.output(TangentialCombine(numBases(), deg, adapt, i));
  }
  
 protected:
  void computePartPoints(f64 sigma, std::vector<f64> &partPoints) {
    i32 resolution = 400;
    f64 fullRange = 20;
    f64 halfRange = fullRange/2;
    std::valarray<f64> partialSums(resolution);
    
    f64 sum = 0;
    for (i32 i = 0; i < resolution; ++i) {
      sum += Gaussian(fullRange*i/resolution - halfRange, sigma);
      partialSums[i] = sum;
    }
    partialSums /= sum;
    
    i32 n = i32(partPoints.size()) + 1;
    i32 partIndx = 0;
    f64 target = (partIndx + 1.)/n;
    for (i32 i = 0; i < resolution - 1; ++i) {
      if (partialSums[i] <= target && partialSums[i + 1] > target) {
        f64 x0 = fullRange*(i + 0.5)/resolution - halfRange;
        f64 x1 = fullRange*(i + 1.5)/resolution - halfRange;
        f64 y0 = partialSums[i] - target;
        f64 y1 = partialSums[i + 1] - target;
        partPoints[partIndx++] = (x1*y0 - x0*y1)/(y0 - y1);
        if (partIndx == n) break;
        target = (partIndx + 1.)/n;
      }
    }
  }
};

}

#endif
