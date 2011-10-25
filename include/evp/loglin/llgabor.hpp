#ifndef EVP_LOGLIN_LLGABOR_H
#define EVP_LOGLIN_LLGABOR_H

#include <utility>

#include "evp/util/ndarray.hpp"

namespace evp {
using namespace clip;

class LLGabor {
  typedef std::pair<ImageData,ImageData> LLConditionKernelPair;
//  typedef std::pair<SparseImageBuffer,SparseImageBuffer> LLConditionKernBufPair;
  
  f64 theta_, thetaStep_, wavelength_, phase_, phaseStep_, sigma_, aspect_;
  f32 degree_;
  bool adapt_;
  
  ImageData baseKern_;
//  ImageBuffer baseBuf_;
  
  std::vector<LLConditionKernelPair> condKernels_;
//  std::vector<LLConditionKernBufPair> condBufs_;
 
 public:
  LLGabor(f64 theta, f64 thetaStep,
          f64 wavelength, f64 phase, f64 phaseStep,
          f64 sigma, f64 aspect,
          f32 degree, bool adapt)
  : theta_(theta), thetaStep_(thetaStep),
    wavelength_(wavelength), phase_(phase), phaseStep_(phaseStep),
    sigma_(sigma), aspect_(aspect),
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
    
    baseKern_ = ImageData(kwidth, kheight);
    LLConditionKernelPair thetaPair(baseKern_.clone(), baseKern_.clone());
    LLConditionKernelPair phasePair(baseKern_.clone(), baseKern_.clone());
    condKernels_.push_back(thetaPair);
    condKernels_.push_back(phasePair);
    
    i32 halfWidth = kwidth/2;
    i32 halfHeight = kheight/2;
    
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
        f64 p1 = phase - phaseStep/2;
        f64 p2 = phase + phaseStep/2;
        
        f64 val = Gabor(xp, yp, wavelength, phase, sigma, aspect);
        f64 dt1 = DGaborDTheta(xp1, yp1, wavelength, phase, sigma, aspect);
        f64 dt2 = DGaborDTheta(xp2, yp2, wavelength, phase, sigma, aspect);
        f64 dp1 = DGaborDPhase(xp, yp, wavelength, p1, sigma, aspect);
        f64 dp2 = DGaborDPhase(xp, yp, wavelength, p2, sigma, aspect);
        
        baseKern_(xi, yi) = val;
        condKernels_[0].first(xi, yi) = dt1;
        condKernels_[0].second(xi, yi) = -dt2;
        condKernels_[1].first(xi, yi) = dp1;
        condKernels_[1].second(xi, yi) = -dp2;
      }
    }
    
    baseKern_.balance().normalize();
    for (i32 i = 0; i < i32(condKernels_.size()); ++i) {
      condKernels_[i].first.balance().normalize();
      condKernels_[i].second.balance().normalize();
    }
  }
  
  ImageBuffer apply(const ImageBuffer& input, ImageBuffer output) {
    ImBufList stack;
    PushAdaptor pusher(stack);
    PopAdaptor popper(stack);
    
    pusher.output(Filter(input, baseKern_));
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
