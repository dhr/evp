#pragma once
#ifndef EVP_UTIL_GABOR_H
#define EVP_UTIL_GABOR_H

#include <algorithm>

#include <clip.hpp>

#include "evp/util/gaussian.hpp"

namespace evp {
using namespace clip;

inline f64 Gabor(f64 x, f64 y, f64 w, f64 p, f64 s, f64 a) {
  return Gaussian(x, s)*Gaussian(y, s/a)*sin(2*M_PI*y/w + p);
}

inline f64 DGaborDTheta(f64 x, f64 y, f64 w, f64 p, f64 s, f64 a) {
  f64 gauss = Gaussian(x, s)*Gaussian(y, s/a);
  f64 off = 2*M_PI*y/w + p;
  return -gauss*x*(2*M_PI*cos(off)/w - (a*a - 1)*y*sin(off)/s/s);
}

inline f64 DGaborDPhase(f64 x, f64 y, f64 w, f64 p, f64 s, f64 a) {
  f64 gauss = Gaussian(x, s)*Gaussian(y, s/a);
  f64 off = 2*M_PI*y/w + p;
  return gauss*cos(off);
}

inline ImageData MakeGabor(f64 theta,
                           f64 wavelength,
                           f64 phase,
                           f64 sigma,
                           f64 aspect,
                           i32 kwidth = 0,
                           i32 kheight = 0) {
  f64 sinTheta = sin(theta);
  f64 cosTheta = cos(theta);
  
  if (kwidth == 0) {
    kwidth = unsigned(floor(std::max(fabs(6*sigma*cosTheta),
                                     fabs(6*sigma/aspect*sinTheta))));
  }
  
  if (kwidth%2 == 0) kwidth++;
  
  if (kheight == 0) {
    kheight = unsigned(floor(std::max(fabs(6*sigma/aspect*cosTheta),
                                      fabs(6*sigma*sinTheta))));
  }
  
  if (kheight%2 == 0) kheight++;
  
  ImageData kernel(kwidth, kheight);
  
  i32 halfWidth = kwidth/2;
  i32 halfHeight = kheight/2;
  
  i32 y = -halfHeight;
  for (i32 yi = 0; yi < kheight; ++y, ++yi) {
    i32 x = -halfWidth;
    for (i32 xi = 0; xi < kwidth; ++x, ++xi) {
      f64 xp = cosTheta*x + sinTheta*y;
      f64 yp = cosTheta*y - sinTheta*x;
      kernel(xi, yi) = Gabor(xp, yp, wavelength, phase, sigma, aspect);
    }
  }
  
  return kernel.balance().normalize();
}

}

#endif
