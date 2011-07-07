#pragma once
#ifndef EVP_UTIL_TICTOC_H
#define EVP_UTIL_TICTOC_H

namespace evp {
using namespace clip;
  static long ticStart;
}

#ifndef _WIN32
#include <sys/time.h>

namespace evp {
using namespace clip;

inline void tic() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  ticStart = tv.tv_sec*1000000 + tv.tv_usec;
}

inline long toc() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  long ticEnd = tv.tv_sec*1000000 + tv.tv_usec;
  return ticEnd - ticStart;
}

}

#else
#include <Windows.h>

namespace evp {
using namespace clip;

inline void tic() {
  ticStart = GetTickCount();
}

inline long toc() {
  return (GetTickCount() - ticStart)*1000;
}

}

#endif

#endif
