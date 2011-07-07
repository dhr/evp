#pragma once
#ifndef EVP_UTIL_MATHUTIL_H
#define EVP_UTIL_MATHUTIL_H

#include <cmath>

namespace evp {
using namespace clip;

#ifndef M_PI
#define M_PI 3.141592653589793238462643
#endif

template<typename T>
inline T sign(T val) { return val > 0 ? 1 : val < 0 ? -1 : 0; }

#ifndef isnan
inline bool isnan(f32 val) { return val != val; }
inline bool isnan(f64 val) { return val != val; }
#endif

inline f64 cmod(f64 val, f64 mod = 2*M_PI) {
  f64 result = fmod(val, mod);
  result += result < 0 ? mod : 0;
  
  f64 halfMod = mod/2;
  if (result < halfMod) return result;
  if (result > halfMod) return result - mod;
  return val > 0 ? result : -result;
}

template<typename T>
inline T clamp(T val, T min, T max) {
  return std::max(std::min(val, max), min);
}

}

#endif
