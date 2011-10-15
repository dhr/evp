#pragma once
#ifndef EVP_EVP_H
#define EVP_EVP_H

#include <clip.hpp>

#include "evp/kernels.hpp"
#include "evp/loglin/llinitops.hpp"
#include "evp/loglin/llinitopparams.hpp"
#include "evp/curve/lineedgelocatorop.hpp"
#include "evp/curve/relaxcurveop.hpp"
#include "evp/curve/relaxcurveopparams.hpp"
#include "evp/curve/suppresslineedgesop.hpp"
#include "evp/flow/flowinitops.hpp"
#include "evp/flow/flowinitopparams.hpp"
#include "evp/flow/jitteredflowinitops.hpp"
#include "evp/flow/relaxflowop.hpp"
#include "evp/flow/relaxflowopparams.hpp"
#include "evp/flow/uniforminhflowop.hpp"
#include "evp/util/gabor.hpp"
#include "evp/util/monitorable.hpp"
#include "evp/util/ndarray.hpp"

namespace evp {
namespace detail {

struct EVPInitializer {
  static void init() {
    AddProgram("loglin", LogLinKernels());
    AddProgram("curve", CurveKernels());
    AddProgram("flow", FlowKernels());
  }

 public:
  EVPInitializer() {
    static bool hasRun = false;
    if (!hasRun) {
      hasRun = true;
      clip::AddInitClient(EVPInitializer::init);
    }
  }
};

static EVPInitializer evpInitializer;

}
}

#endif
