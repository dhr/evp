#pragma once
#ifndef EVP_UTIL_GABOR_H
#define EVP_UTIL_GABOR_H

#include <algorithm>

#include <clip.hpp>

namespace evp {
using namespace clip;

ImageData MakeGabor(f64 theta,
                    f64 wavelength,
                    f64 phase,
                    f64 sigma,
                    f64 aspect,
                    i32 kwidth = 0,
                    i32 kheight = 0) {
  f64 sinTheta = sin(theta);
  f64 cosTheta = cos(theta);
  
  f64 xsigma = sigma;
  f64 ysigma = sigma/aspect;
  
  if (kwidth == 0) {
    kwidth = unsigned(floor(std::max(fabs(6*xsigma*cosTheta),
                                     fabs(6*ysigma*sinTheta))));
  }
  
  if (kwidth%2 == 0) kwidth++;
  
  if (kheight == 0) {
    kheight = unsigned(floor(std::max(fabs(6*ysigma*cosTheta),
                                      fabs(6*xsigma*sinTheta))));
  }
  
  if (kheight%2 == 0) kheight++;
  
  ImageData kernel(kwidth, kheight);
  
  i32 halfWidth = kwidth/2;
  i32 halfHeight = kheight/2;
  
  f64 xfreq2 = 1/(2*xsigma*xsigma);
  f64 yfreq2 = 1/(2*ysigma*ysigma);
  
  i32 y = -halfHeight;
  for (i32 yi = 0; yi < kheight; ++y, ++yi) {
    i32 x = -halfWidth;
    for (i32 xi = 0; xi < kwidth; ++x, ++xi) {
      f64 xp = cosTheta*x + sinTheta*y;
      f64 yp = cosTheta*y - sinTheta*x;
      f64 val = exp(-(xp*xp*xfreq2 + yp*yp*yfreq2));
      kernel(xi, yi) = val*sin(2*M_PI*yp/wavelength + phase);
    }
  }
  
  return kernel.balance().normalize();
}

}

#endif
