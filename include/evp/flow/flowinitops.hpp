#pragma once
#ifndef EVP_FLOW_FLOWINITOPS_H
#define EVP_FLOW_FLOWINITOPS_H

#include <clip.hpp>

#include "evp/flow/flowinitopparams.hpp"
#include "evp/flow/flowtypes.hpp"

namespace evp {
using namespace clip;

class FlowInitOps : public Monitorable {
 public:
  virtual FlowBuffersPtr apply(const ImageBuffer& image) = 0;
};

}

#endif