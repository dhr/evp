#ifndef EVP_CURVE_CURVECOLS_H
#define EVP_CURVE_CURVECOLS_H

#include "evp/util/memutil.hpp"
#include "evp/util/ndarray.hpp"

namespace evp {
using namespace clip;

typedef NDArray<ImageBuffer,2> CurveBuffers;
typedef std::tr1::shared_ptr<CurveBuffers> CurveBuffersPtr;

typedef NDArray<ImageData,2> CurveData;
typedef std::tr1::shared_ptr<CurveData> CurveDataPtr;

}

#endif