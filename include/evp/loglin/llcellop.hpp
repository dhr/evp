#pragma once
#ifndef EVP_LOGLIN_LLCELLOP_H
#define EVP_LOGLIN_LLCELLOP_H

namespace evp {
using namespace clip;

class LLCellOp {
 public:
  virtual ~LLCellOp() {};
  virtual void apply(const ImageBuffer& image, ImageBuffer* output) = 0;
};

typedef std::tr1::shared_ptr<LLCellOp> LLCellOpPtr;

}

#endif
