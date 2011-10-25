#pragma once
#ifndef EVP_UTIL_GAUSSIAN_H
#define EVP_UTIL_GAUSSIAN_H

#include <limits>

#include "evp/util/mathutil.hpp"

namespace evp {
using namespace clip;

inline f64 Gaussian(f64 x, f64 s, bool normalize = true)
{
  f64 val = exp(-x*x/(2*s*s));
  if (normalize) val /= sqrt(2*M_PI)*s;
  return val;
}

inline f64 DGaussian(f64 x, f64 s, i32 n)
{
  switch (n) {
    case 0:
      return Gaussian(x, s);
    case 1:
      return -x/(s*s)*Gaussian(x, s);
    case 2: {
      f64 s2 = s*s;
      return (x*x - s2)/(s2*s2)*Gaussian(x, s);
    }
    case 3: {
      f64 s2 = s*s;
      return x*(3*s2 - x*x)/(s2*s2*s2)*Gaussian(x, s);
    }
    case 4: {
      f64 x2 = x*x;
      f64 s2 = s*s;
      f64 s4 = s2*s2;
      return (3*s4 - 6*s2*x2 + x2*x2)/(s4*s4)*Gaussian(x, s);
    }
    default: // Nothing further is implemented... this'll show 'em!
      return std::numeric_limits<f64>::signaling_NaN();
  }
}

}

#endif
