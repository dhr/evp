#pragma once
#ifndef EVP_LOGLIN_LLOPS_H
#define EVP_LOGLIN_LLOPS_H

#include <cassert>
#include <cmath>

#include <limits>

#include <clip.hpp>

namespace evp {
using namespace clip;

class LLOp {
 protected:
  std::vector<clip::CachedKernel> caches_;
  
 public:
  typedef ImageBuffer result_type;
  
  static const i32 kMinLLArgs = 1;
  static const i32 kMaxLLArgs = 6;
  
  LLOp(std::string baseName) : caches_(kMaxLLArgs - kMinLLArgs + 1) {
    std::stringstream ss;
    for (i32 i = kMinLLArgs; i <= kMaxLLArgs; i++) {
      ss.str("");
      ss << baseName << i;
      caches_[i - kMinLLArgs] = CachedKernel(ss.str());
    }
  }
  
  ImageBuffer operator()(const ImageBuffer& i1,
                         const ImageBuffer& i2,
                         f32 degree,
                         bool adapt,
                         f32 scale,
                         ImageBuffer o) {
    i32 n = 2;
    cl::Kernel& k = caches_[n - kMinLLArgs].get();
    
    i32 i = 0;
    k.setArg(i++, i1.mem());
    k.setArg(i++, i2.mem());
    k.setArg(i++, degree);
    k.setArg(i++, adapt);
    k.setArg(i++, scale);
    k.setArg(i++, o.mem());
    
    Enqueue(k, o);
    
    return o;
  }
  
 ImageBuffer operator()(const ImageBuffer& i1,
                        const ImageBuffer& i2,
                        f32 degree,
                        bool adapt,
                        f32 scale) {
    return operator()(i1, i2, degree, adapt, scale, ~i1);
  }
  
 ImageBuffer operator()(const ImageBuffer& i1,
                        const ImageBuffer& i2,
                        const ImageBuffer& i3,
                        f32 degree,
                        bool adapt,
                        f32 scale,
                        ImageBuffer o) {
    i32 n = 3;
    cl::Kernel& k = caches_[n - kMinLLArgs].get();
    
    i32 i = 0;
    k.setArg(i++, i1.mem());
    k.setArg(i++, i2.mem());
    k.setArg(i++, i3.mem());
    k.setArg(i++, degree);
    k.setArg(i++, adapt);
    k.setArg(i++, scale);
    k.setArg(i++, o.mem());
    
    Enqueue(k, o);
    
    return o;
  }
  
  ImageBuffer operator()(const ImageBuffer& i1,
                         const ImageBuffer& i2,
                         const ImageBuffer& i3,
                         f32 degree,
                         bool adapt,
                         f32 scale) {
    return operator()(i1, i2, i3, degree, adapt, scale, ~i1);
  }
  
  ImageBuffer operator()(const ImageBuffer& i1,
                         const ImageBuffer& i2,
                         const ImageBuffer& i3,
                         const ImageBuffer& i4,
                         f32 degree,
                         bool adapt,
                         f32 scale,
                         ImageBuffer o) {
    i32 n = 4;
    cl::Kernel& k = caches_[n - kMinLLArgs].get();
    
    i32 i = 0;
    k.setArg(i++, i1.mem());
    k.setArg(i++, i2.mem());
    k.setArg(i++, i3.mem());
    k.setArg(i++, i4.mem());
    k.setArg(i++, degree);
    k.setArg(i++, adapt);
    k.setArg(i++, scale);
    k.setArg(i++, o.mem());
    
    Enqueue(k, o);
    
    return o;
  }
  
  ImageBuffer operator()(const ImageBuffer& i1,
                         const ImageBuffer& i2,
                         const ImageBuffer& i3,
                         const ImageBuffer& i4,
                         f32 degree,
                         bool adapt,
                         f32 scale) {
    return operator()(i1, i2, i3, i4, degree, adapt, scale, ~i1);
  }
  
  ImageBuffer operator()(const ImageBuffer& i1,
                         const ImageBuffer& i2,
                         const ImageBuffer& i3,
                         const ImageBuffer& i4,
                         const ImageBuffer& i5,
                         f32 degree,
                         bool adapt,
                         f32 scale,
                         ImageBuffer o) {
    i32 n = 5;
    cl::Kernel& k = caches_[n - kMinLLArgs].get();
    
    i32 i = 0;
    k.setArg(i++, i1.mem());
    k.setArg(i++, i2.mem());
    k.setArg(i++, i3.mem());
    k.setArg(i++, i4.mem());
    k.setArg(i++, i5.mem());
    k.setArg(i++, degree);
    k.setArg(i++, adapt);
    k.setArg(i++, scale);
    k.setArg(i++, o.mem());
    
    Enqueue(k, o);
    
    return o;
  }
  
  ImageBuffer operator()(const ImageBuffer& i1,
                         const ImageBuffer& i2,
                         const ImageBuffer& i3,
                         const ImageBuffer& i4,
                         const ImageBuffer& i5,
                         f32 degree,
                         bool adapt,
                         f32 scale) {
    return operator()(i1, i2, i3, i4, i5, degree, adapt, scale, ~i1);
  }
  
  ImageBuffer operator()(const ImageBuffer& i1,
                         const ImageBuffer& i2,
                         const ImageBuffer& i3,
                         const ImageBuffer& i4,
                         const ImageBuffer& i5,
                         const ImageBuffer& i6,
                         f32 degree,
                         bool adapt,
                         f32 scale,
                         ImageBuffer o) {
    i32 n = 6;
    cl::Kernel& k = caches_[n - kMinLLArgs].get();
    
    i32 i = 0;
    k.setArg(i++, i1.mem());
    k.setArg(i++, i2.mem());
    k.setArg(i++, i3.mem());
    k.setArg(i++, i4.mem());
    k.setArg(i++, i5.mem());
    k.setArg(i++, i6.mem());
    k.setArg(i++, degree);
    k.setArg(i++, adapt);
    k.setArg(i++, scale);
    k.setArg(i++, o.mem());
    
    Enqueue(k, o);
    
    return o;
  }
  
  ImageBuffer operator()(const ImageBuffer& i1,
                         const ImageBuffer& i2,
                         const ImageBuffer& i3,
                         const ImageBuffer& i4,
                         const ImageBuffer& i5,
                         const ImageBuffer& i6,
                         f32 degree,
                         bool adapt,
                         f32 scale) {
    return operator()(i1, i2, i3, i4, i5, i6, degree, adapt, scale, ~i1);
  }
};

struct LLAndOp : public LLOp { LLAndOp() : LLOp("lland") {} };
static LLAndOp LLAnd;

struct LLOrOp : public LLOp { LLOrOp() : LLOp("llor") {} };
static LLOrOp LLOr;

inline ImageBuffer
LLMerge(LLOp op, i32 n, f32 degree, bool adapt, f32 scale,
        InputAdaptor& input, ImageBuffer output) {
  assert(n >= LLOp::kMinLLArgs && "Too few arguments passed to LLMerge");
  assert(n <= LLOp::kMaxLLArgs && "Too many arguments passed to LLMerge");
  
  switch (n) {
    case 1:
      input.next().copyInto(output);
      return output;
      
    case 2:
      return op(input.next(), input.next(),
                degree, adapt, scale, output);
    
    case 3:
      return op(input.next(), input.next(), input.next(),
                degree, adapt, scale, output);
    
    case 4:
      return op(input.next(), input.next(),  input.next(), input.next(),
                degree, adapt, scale, output);
      
    case 5:
      return op(input.next(), input.next(), input.next(),
                input.next(), input.next(),
                degree, adapt, scale, output);
    
    case 6:
      return op(input.next(), input.next(), input.next(),
                input.next(), input.next(), input.next(),
                degree, adapt, scale, output);
    
    default:
      assert(false && "Unexpected number of arguments in LLMerge!!!");
      return ImageBuffer(); // Won't be reached, just silencing warnings
  }
}

inline ImageBuffer
LLMerge(LLOp op, i32 n, f32 degree, bool adapt, f32 scale,
        InputAdaptor& input) {
  return LLMerge(op, n, degree, adapt, scale, input, ~input.peek());
}

}

#endif
