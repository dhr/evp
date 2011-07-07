#pragma once
#ifndef EVP_CURVE_CURVECOMPATFUNCS_H
#define EVP_CURVE_CURVECOMPATFUNCS_H

#include <tr1/array>

#include "evp/util/mathutil.hpp"

namespace evp {
using namespace clip;

struct ProjDiffs {
  f64 xy;
  f64 theta;
  f64 kappa;
  f64 transport;
};

inline void ProjectionDifference(const std::tr1::array<f64, 4> &i,
                                 const std::tr1::array<f64, 4> &j,
                                 i32 numPis, ProjDiffs &diffs) {
  f64 cosTi = cos(i[2]);
  f64 sinTi = sin(i[2]);
  f64 xjRot = cosTi*(j[0] - i[0]) + sinTi*(j[1] - i[1]);
  f64 yjRot = cosTi*(j[1] - i[1]) - sinTi*(j[0] - i[0]);
  f64 tjRot = j[2] - i[2];
  f64 cosTjRot = cos(tjRot);
  f64 sinTjRot = sin(tjRot);
  f64 xProj, yProj;
  
  if (j[3] == 0) {
    diffs.theta = cmod(tjRot);
    diffs.transport = cosTjRot*xjRot + sinTjRot*yjRot;
    xProj = xjRot - cosTjRot*diffs.transport;
    yProj = yjRot - sinTjRot*diffs.transport;
  }
  else {
    f64 xjRotCenter = xjRot - sinTjRot/j[3];
    f64 yjRotCenter = yjRot + cosTjRot/j[3];
    f64 jDist = sqrt(xjRotCenter*xjRotCenter + yjRotCenter*yjRotCenter);
    f64 angle = jDist != 0 ? atan2(-yjRotCenter, -xjRotCenter) : 0;
    f64 rad = fabs(1/j[3]);
    
    // Equivalent to xjRotCenter + rad*cos(angle), but more accurate
    f64 temp = 1.f - rad/jDist;
    xProj = jDist != 0 ? xjRotCenter*temp : xjRotCenter + rad;
    yProj = jDist != 0 ? yjRotCenter*temp : yjRotCenter;
    
    angle += j[3] > 0 ? M_PI/2 : -M_PI/2;
    diffs.theta = cmod(angle);
    diffs.transport = -cmod(angle - tjRot)/j[3];
  }
  
  diffs.xy = sqrt(xProj*xProj + yProj*yProj)*(yProj >= 0 ? 1 : -1);
  
  if (numPis == 1 && fabs(diffs.theta) > M_PI/2) {
    diffs.theta += diffs.theta > 0 ? -M_PI : M_PI;
    diffs.kappa = -j[3] - i[3];
    diffs.transport *= -1;
  }
  else {
    diffs.kappa = j[3] - i[3];
  }
}

}

#endif
