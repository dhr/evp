#pragma once
#ifndef EVP_FLOW_FLOWMODEL_H
#define EVP_FLOW_FLOWMODEL_H

#include <limits.h>

#include "evp/util/mathutil.hpp"
#include "evp/flow/relaxflowopparams.hpp"

namespace evp {
using namespace clip;

class FlowModel {
  f64 centerX_, centerY_, theta_, kt_, kn_;
  f64 del2_;
  
  void updateDel2() {
    del2_ = kt_*kt_ + kn_*kn_;
  }
  
 public:
  FlowModel() {}
  
  FlowModel(f64 centerX, f64 centerY,
            f64 theta, f64 kt, f64 kn)
  : centerX_(centerX), centerY_(centerY),
    theta_(theta), kt_(kt), kn_(kn)
  {
    updateDel2();
  }
  
  bool gradientAt(f64 x, f64 y, f64 *gx, f64 *gy) {
    x -= centerX_;
    y -= centerY_;
    f64 gxn = -del2_*y + kt_*cos(theta_) - kn_*sin(theta_);
    f64 gyn = del2_*x + kt_*sin(theta_) + kn_*cos(theta_);
    f64 gd = 1 + del2_*(x*x + y*y) +
                2*(kn_*x - kt_*y)*cos(theta_) +
                2*(kt_*x + kn_*y)*sin(theta_);
    
    if (gd == 0.0) return false;
    
    *gx = gxn/gd;
    *gy = gyn/gd;
    
    return true;
  }
  
  f64 thetaAt(f64 x, f64 y) {
    f64 rx = cos(theta_)*(x - centerX_) + sin(theta_)*(y - centerY_);
    f64 ry = -sin(theta_)*(x - centerX_) + cos(theta_)*(y - centerY_);
    f64 vx = 1 + kn_*rx - kt_*ry;
    f64 vy = kt_*rx + kn_*ry;
    
    f64 theta = theta_ + (vx == 0.0 ? M_PI/2.0*sign(vy) : atan2(vy, vx));
    return cmod(theta);
  }
  
  f64 ktAt(f64 x, f64 y) {
    f64 gx, gy;
    
    f64 theta = thetaAt(x, y);
    if (!gradientAt(x, y, &gx, &gy))
      return std::numeric_limits<f64>::quiet_NaN();
    
    return gx*cos(theta) + gy*sin(theta);
  }
  
  f64 knAt(f64 x, f64 y) {
    f64 gx, gy;
    
    f64 theta = thetaAt(x, y);
    if (!gradientAt(x, y, &gx, &gy))
      return std::numeric_limits<f64>::quiet_NaN();
    
    return -gx*sin(theta) + gy*cos(theta);
  }
  
  bool valuesAt(f64 x, f64 y, f64 *theta, f64 *kt, f64 *kn) {
    *theta = thetaAt(x, y);
    
    f64 gx, gy;
    if (!gradientAt(x, y, &gx, &gy)) {
      *kt = std::numeric_limits<f64>::quiet_NaN();
      *kn = *kt;
      
      return false;
    }
    
    f64 cosTheta = cos(*theta);
    f64 sinTheta = sin(*theta);
    
    *kt = gx*cosTheta + gy*sinTheta;
    *kn = -gx*sinTheta + gy*cosTheta;
    
    return true;
  }
  
  f64 centerX() { return centerX_; }
  f64 centerY() { return centerY_; }
  f64 theta() { return theta_; }
  f64 kt() { return kt_; }
  f64 kn() { return kn_; }
  
  void setCenterX(f64 centerX) { centerX_ = centerX; }
  void setCenterY(f64 centerY) { centerY_ = centerY; }
  void setTheta(f64 theta) { theta_ = theta; }
  void setKt(f64 kt) { kt_ = kt; updateDel2(); }
  void setKn(f64 kn) { kn_ = kn; updateDel2(); }
  
  void set(f64 centerX, f64 centerY,
           f64 theta, f64 kt, f64 kn) {
    centerX_ = centerX;
    centerY_ = centerY;
    theta_ = theta;
    kt_ = kt;
    kn_ = kn;
    updateDel2();
  }
};

}

#endif
