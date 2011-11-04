#ifndef EVP_FLOW_LLFLOWINITOPS_H
#define EVP_FLOW_LLFLOWINITOPS_H

#include "evp/loglin/llgabor.hpp"

namespace evp {
using namespace clip;
  
class LLFlowInitOps : public Monitorable {
  FlowInitOpParams params_;
  std::vector<LLGabor> llgabors_[4];
  
  f32 wavelength_, sigma_, aspect_, degree_;
  bool adapt_;

 public:
  LLFlowInitOps(FlowInitOpParams& params)
  : params_(params), wavelength_(8), sigma_(2),
    aspect_(1), degree_(32), adapt_(true)
  {
    i32 nt = params.numOrientations;
    f32 dt = params.orientationStep;
    f32 dp = M_PI/2;
    f32 ds = 0.5;
    f32 w = wavelength_, s = sigma_, a = aspect_, deg = degree_;
    bool ad = adapt_;
    
    for (i32 i = 0; i < nt; ++i) {
      llgabors_[0].push_back(LLGabor(i*dt, dt, w, 0, dp, s, ds, a, deg, ad));
      llgabors_[1].push_back(LLGabor(i*dt, dt, w, dp, dp, s, ds, a, deg, ad));
      llgabors_[2].push_back(LLGabor(i*dt, dt, w, 2*dp, dp, s, ds, a, deg, ad));
      llgabors_[3].push_back(LLGabor(i*dt, dt, w, 3*dp, dp, s, ds, a, deg, ad));
    }
  }
  
  FlowBuffersPtr apply(const ImageBuffer& image) {
    FlowBuffersPtr outputPtr(new FlowBuffers(params_.numOrientations,
                                             params_.numCurvatures,
                                             params_.numCurvatures));
    FlowBuffers& output = *outputPtr;
    
    ImBufList stack;
    PopAdaptor popper(stack);
    PushAdaptor pusher(stack);
    
    for (i32 i = 0; i < i32(llgabors_[0].size()); ++i) {
//      pusher.output(llgabors_[0][i].apply(image));
      pusher.output(llgabors_[1][i].apply(image));
//      pusher.output(llgabors_[2][i].apply(image));
      pusher.output(llgabors_[3][i].apply(image));
      
      ImageBuffer temp = LLMerge(LLOr, 2, degree_, false, 1, popper);
      Bound(temp, temp);
      
      output(i, 0, 0) = temp;
      for (i32 ktii = 0; ktii < params_.numCurvatures; ++ktii) {
        for (i32 knii = 0; knii < params_.numCurvatures; ++knii) {
          if (!ktii && !knii) continue;
          
          NDIndex<3> target(i, ktii, knii);
          output[target] = temp.clone();
        }
      }
      
      setProgress(f32(i + 1)/llgabors_[0].size());
    }
    
    return outputPtr;
  }
};
  
}

#endif
