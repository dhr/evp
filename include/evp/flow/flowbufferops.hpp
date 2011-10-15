#ifndef EVP_FLOW_FLOWBUFFEROPS_H
#define EVP_FLOW_FLOWBUFFEROPS_H

namespace evp {
using namespace clip;

struct RescaleOp : public BasicOp {
  typedef ImageBuffer result_type;
  
  RescaleOp() : BasicOp("rescale") {}
  
  inline ImageBuffer operator()(const ImageBuffer& i1,
                                f32 min, f32 max,
                                f32 targetMin, f32 targetMax,
                                bool filter,
                                ImageBuffer o) {
    cl::Kernel &kernel = cache_.get();
    kernel.setArg(0, i1.mem());
    kernel.setArg(1, min);
    kernel.setArg(2, max);
    kernel.setArg(3, targetMin);
    kernel.setArg(4, targetMax);
    kernel.setArg(5, filter ? 1 : 0);
    kernel.setArg(6, o.mem());
    Enqueue(kernel, o);
    return o;
  }
  
  inline ImageBuffer operator()(const ImageBuffer& i1,
                                f32 min, f32 max,
                                f32 targetMin, f32 targetMax,
                                bool filter) {
    return operator()(i1, min, max, targetMin, targetMax, filter, ~i1);
  }
};
static RescaleOp Rescale;

struct Grad2PolarOp : public BasicOp {
  typedef void result_type;
  
  Grad2PolarOp() : BasicOp("grad2polar") {}
  
  inline void operator()(const ImageBuffer& xs, const ImageBuffer& ys,
                         ImageBuffer rs, ImageBuffer thetas) {
    cl::Kernel &kernel = cache_.get();
    kernel.setArg(0, xs.mem());
    kernel.setArg(1, ys.mem());
    kernel.setArg(2, rs.mem());
    kernel.setArg(3, thetas.mem());
    Enqueue(kernel, thetas);
  }
};
static Grad2PolarOp Grad2Polar;

struct UnitVectorizeOp : public BasicOp {
  typedef void result_type;
  
  UnitVectorizeOp() : BasicOp("unitvec") {}
  
  inline void operator()(const ImageBuffer& angles,
                         ImageBuffer us, ImageBuffer vs) {
    cl::Kernel &kernel = cache_.get();
    kernel.setArg(0, angles.mem());
    kernel.setArg(1, us.mem());
    kernel.setArg(2, vs.mem());
    Enqueue(kernel, vs);
  }
};
static UnitVectorizeOp UnitVectorize;

}

#endif
