#ifndef EVP_LOGLIN_LLGABOR_H
#define EVP_LOGLIN_LLGABOR_H

#include <cmath>
#include <utility>

#include "evp/util/ndarray.hpp"

namespace evp {
using namespace clip;

class LLGabor {
  typedef std::pair<ImageData,ImageData> LLConditionKernelPair;
//  typedef std::pair<SparseImageBuffer,SparseImageBuffer> LLConditionKernBufPair;
  
  
  f64 theta_, thetaStep_;
  i32 nTangentials_;
  f64 wavelength_, phase_, phaseStep_;
  f64 sigma_, sigmaStep_, aspect_;
  f32 degree_;
  bool adapt_;
  
  std::vector<SparseImageData> baseKerns_;
//  ImageBuffer baseBuf_;
  
  std::vector<LLConditionKernelPair> condKernels_;
//  std::vector<LLConditionKernBufPair> condBufs_;
 
 public:
  LLGabor(f64 theta, f64 thetaStep,
//          i32 nTangentials,
          f64 wavelength, f64 phase, f64 phaseStep,
          f64 sigma, f64 sigmaStep, f64 aspect,
          f32 degree, bool adapt)
  : theta_(theta), thetaStep_(thetaStep), nTangentials_(4),
    wavelength_(wavelength), phase_(phase), phaseStep_(phaseStep),
    sigma_(sigma), sigmaStep_(sigmaStep), aspect_(aspect),
    degree_(degree), adapt_(adapt)
  {
    i32 kwidth, kheight;
    
    f64 sinTheta = sin(theta);
    f64 cosTheta = cos(theta);
    f64 sinTheta1 = sin(theta - thetaStep/2);
    f64 cosTheta1 = cos(theta - thetaStep/2);
    f64 sinTheta2 = sin(theta + thetaStep/2);
    f64 cosTheta2 = cos(theta + thetaStep/2);
    
    kwidth = unsigned(floor(std::max(fabs(6*sigma*cosTheta),
                                     fabs(6*sigma/aspect*sinTheta))));
    kheight = unsigned(floor(std::max(fabs(6*sigma/aspect*cosTheta),
                                      fabs(6*sigma*sinTheta))));
    
    if (kwidth%2 == 0) kwidth++;
    if (kheight%2 == 0) kheight++;
    
    std::vector<ImageData> baseTemp;
    for (i32 tani = 0; tani < nTangentials_; ++tani) {
      baseTemp.push_back(ImageData(kwidth, kheight));
    }
    
    ImageData tmp = ImageData(kwidth, kheight);
    LLConditionKernelPair thetaPair(tmp.clone(), tmp.clone());
    LLConditionKernelPair phasePair(tmp.clone(), tmp.clone());
    LLConditionKernelPair scalePair(tmp.clone(), tmp.clone());
    condKernels_.push_back(thetaPair);
    condKernels_.push_back(phasePair);
    condKernels_.push_back(scalePair);
    
    i32 halfWidth = kwidth/2;
    i32 halfHeight = kheight/2;
    
    f64 p1 = phase - phaseStep/2;
    f64 p2 = phase + phaseStep/2;
    f64 s1 = sigma - sigmaStep/2;
    f64 s2 = sigma + sigmaStep/2;
    
    i32 y = -halfHeight;
    for (i32 yi = 0; yi < kheight; ++y, ++yi) {
      i32 x = -halfWidth;
      for (i32 xi = 0; xi < kwidth; ++x, ++xi) {
        f64 xp = cosTheta*x + sinTheta*y;
        f64 yp = cosTheta*y - sinTheta*x;
        f64 xp1 = cosTheta1*x + sinTheta1*y;
        f64 yp1 = cosTheta1*y - sinTheta1*x;
        f64 xp2 = cosTheta2*x + sinTheta2*y;
        f64 yp2 = cosTheta2*y - sinTheta2*x;
        
        f64 val = Gabor(xp, yp, wavelength, phase, sigma, aspect);
        f64 dt1 = DGaborDTheta(xp1, yp1, wavelength, phase, sigma, aspect);
        f64 dt2 = DGaborDTheta(xp2, yp2, wavelength, phase, sigma, aspect);
        f64 dp1 = DGaborDPhase(xp, yp, wavelength, p1, sigma, aspect);
        f64 dp2 = DGaborDPhase(xp, yp, wavelength, p2, sigma, aspect);
        f64 ds1 = DGaborDScale(xp, yp, wavelength, phase, s1, aspect);
        f64 ds2 = DGaborDScale(xp, yp, wavelength, phase, s2, aspect);
        
        for (i32 tani = 0; tani < nTangentials_; ++tani) {
          f64 cdf = 0.5*(1 + erf(xp/sqrt(2)/sigma));
          bool use = cdf > f32(tani)/nTangentials_ &&
                     cdf < f32(tani + 1)/nTangentials_;
          baseTemp[tani](xi, yi) = use ? val : 0;
        }
        
        condKernels_[0].first(xi, yi) = dt1;
        condKernels_[0].second(xi, yi) = -dt2;
        condKernels_[1].first(xi, yi) = dp1;
        condKernels_[1].second(xi, yi) = -dp2;
        condKernels_[2].first(xi, yi) = ds1;
        condKernels_[2].second(xi, yi) = -ds2;
      }
    }
    
    for (i32 tani = 0; tani < nTangentials_; ++tani) {
      baseTemp[tani].balance().normalize();
      baseTemp[tani].data() /= nTangentials_;
      baseKerns_.push_back(SparseImageData(baseTemp[tani]));
    }
    
    for (i32 i = 0; i < i32(condKernels_.size()); ++i) {
      condKernels_[i].first.balance().normalize();
      condKernels_[i].second.balance().normalize();
    }
  }
  
  ImageBuffer apply(const ImageBuffer& input, ImageBuffer output) {
    ImBufList stack;
    PushAdaptor pusher(stack);
    PopAdaptor popper(stack);
    
    std::vector<ImageBuffer> tangentials;
    for (i32 tani = 0; tani < nTangentials_; ++tani)
      tangentials.push_back(Filter(input, baseKerns_[tani]));
    pusher.output(LLAnd(tangentials[0], tangentials[2], degree_, adapt_, 1.f));
    pusher.output(LLAnd(tangentials[1], tangentials[3], degree_, adapt_, 1.f));
    pusher.output(LLAnd(tangentials[0], tangentials[3], degree_, adapt_, 1.f));
    pusher.output(LLMerge(LLOr, 3, degree_, adapt_, 1.f, popper));
    
    for (i32 i = 0; i < i32(condKernels_.size()); ++i) {
      ImageBuffer f1 = Filter(input, condKernels_[i].first);
      ImageBuffer f2 = Filter(input, condKernels_[i].second);
      pusher.output(LLAnd(f1, f2, degree_, adapt_, 0.5f));
    }
    LLAnd(popper.next(), popper.next(), degree_, adapt_, 0.5f, output);
    
    return output;
  }
  
  ImageBuffer apply(ImageBuffer input) {
    return apply(input, ~input);
  }
};

}

#endif
