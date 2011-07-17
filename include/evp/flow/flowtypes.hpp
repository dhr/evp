#ifndef EVP_FLOW_FLOWTYPES_H
#define EVP_FLOW_FLOWTYPES_H

#include "evp/util/memutil.hpp"
#include "evp/util/ndarray.hpp"

namespace evp {
using namespace clip;

typedef NDArray<ImageBuffer,3> FlowBuffers;
typedef std::tr1::shared_ptr<FlowBuffers> FlowBuffersPtr;

typedef NDArray<ImageData,3> FlowData;
typedef std::tr1::shared_ptr<FlowData> FlowDataPtr;

}

#endif