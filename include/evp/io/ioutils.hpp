#pragma once
#ifndef EVP_IO_IOUTILS_H
#define EVP_IO_IOUTILS_H

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <sstream>
#include <string>
#include <vector>

#include <clip.hpp>

#include "evp/io/pdfwriter.hpp"
#include "evp/util/mathutil.hpp"
#include "evp/util/ndarray.hpp"

namespace evp {
using namespace clip;

inline void SlurpFile(const std::string& filename,
                      std::string* str,
                      bool isBinary = false) {
  std::ios_base::openmode mode = std::ios::in | std::ios::ate;
  if (isBinary) mode |= std::ios::binary;
  std::ifstream ifs(filename.c_str(), mode);
  std::ifstream::pos_type fileSize = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  str->resize(fileSize);
  ifs.read(&(*str)[0], fileSize);
}

inline void TextualProgressMonitor(f32 progress, i32 total = 35) {
  static f32 last = 0;
  i32 tot = 35;

  if (last == 0) {
    for (i32 i = 0; i < tot; ++i)
      std::cout << "_";
    std::cout << "\n";
  }

  i32 n = i32(progress*tot) - i32(last*tot);
  for (i32 i = 0; i < n; ++i)
    std::cout << '^';

  if (progress == 1) {
    last = 0;
    std::cout << "\n";
  }
  else
    last = progress;

  std::cout.flush();
}

inline void WriteLLColumnsToPDF(const std::string& filename,
                                const CurveData& cols,
                                f32 threshold = 0.01f,
                                f32 darken = 0.3f,
                                f32 length = 1.2) {
  i32 width = cols(0, 0).width();
  i32 height = cols(0, 0).height();
  
  std::ofstream ofs(filename.c_str());
  PDFWriter pdf(ofs, width, height);
  
  i32 no = cols.size(0);
  i32 nk = cols.size(1);
  
  pdf.setLineCapStyle(1);
  pdf.translate(0.5f, 0.5f);
  
  length /= 2;
  for (i32 t = 0; t < no; ++t) {
    f32 o = 2*t*M_PI/no;
    f32 cosDir = cos(o);
    f32 sinDir = sin(o);
    
    for (i32 k = 0; k < nk; ++k) {
      const ImageData &data = cols(t, k);
      
      f32 r = 1.f/(k - nk/2)/2;
      f32 xoff = cosDir*length;
      f32 yoff = sinDir*length;
      
      for (i32 y = 0; y < height; ++y) {
        for (i32 x = 0; x < width; ++x) {
          if (data(x, y) < threshold) continue;
          
          pdf.setStrokeGrayLevel(std::max(1 - powf(data(x, y), darken), 0.f));
          pdf.setLineWidth(0.3);
          
          if (k == nk/2)
            pdf.drawLine(x - xoff, y - yoff, x + xoff, y + yoff);
          else {
            f32 cx = x - r*sinDir;
            f32 cy = y + r*cosDir;
            
            f32 theta1, theta2;
            if (r > 0) {
              theta1 = -M_PI/2 - length/r;
              theta2 = -M_PI/2 + length/r;
            }
            else {
              theta1 = M_PI/2 + length/r;
              theta2 = M_PI/2 - length/r;
            }
            
            pdf.drawArc(cx, cy, fabs(r), theta1 + o, theta2 + o);
          }
          
          pdf.stroke();
        }
      }
    }
  }
  
  pdf.finish();
}

inline void WriteCurveCompatibilitiesToPDF
    (const std::string &filename, const NDArray<ImageData,4> &components,
     bool piSymmetric, f32 length = 0.8f, f32 threshFactor = 0.0001f) {
  NDArray<ImageData,2> compats(components.size(2), components.size(3));
  i32 kernSize = components[0].width();
  
  for (i32 tii = 0; tii < compats.size(0); tii++) {
    for (i32 kii = 0; kii < compats.size(1); kii++) {
      compats(tii, kii) = ImageData(kernSize, kernSize);
      ImageData &compat = compats(tii, kii);
      
      for (i32 ni = 0; ni < components.size(1); ni++) {
        for (i32 tani = 0; tani < components.size(0); tani++) {
          compat.data() += components(tani, ni, tii, kii).data();
        }
      }
    }
  }
  
  std::vector<ImageData> reductions(compats.size(0));
  
  for (i32 tii = 0; tii < compats.size(0); tii++) {
    reductions[tii] = ImageData(kernSize, kernSize);
    
    for (i32 kii = 0; kii < compats.size(1); kii++) {
      reductions[tii].data() += compats(tii, kii).data();
    }
  }
  
  f32 minVal = 0;
  f32 maxVal = 0;
  f32 maxAbsVal = 0;
  
  for (std::vector<ImageData>::iterator it = reductions.begin(),
       end = reductions.end(); it != end; ++ it) {
    minVal = std::min(minVal, it->data().min());
    maxVal = std::max(maxVal, it->data().max());
  }
  maxAbsVal = std::max(maxVal, -minVal);
  
  f32 threshold = threshFactor*maxAbsVal;
  
  std::ofstream ofs(filename.c_str());
  PDFWriter pdf(ofs, kernSize, kernSize);
  
  pdf.setFillGrayLevel(0.5f);
  pdf.drawRect(0, 0, kernSize, kernSize);
  pdf.fill();
  
  pdf.setLineCapStyle(1);
  pdf.setLineWidth(0.1f);
  pdf.translate(0.5f, 0.5f);
  
  length /= 2;
  i32 numPis = 2 - piSymmetric;
  for (i32 y = 0; y < kernSize; y++) {
    for (i32 x = 0; x < kernSize; x++) {
      std::vector<std::pair<f32, int> > lines(reductions.size());
      
      for (i32 tii = 0; tii < i32(lines.size()); tii++) {
        lines[tii] = std::make_pair(fabs(reductions[tii](x, y)), tii);
      }
      
      std::sort(lines.begin(), lines.end());
      
      for (i32 i = 0; i < i32(lines.size()); i++) {
        if (lines[i].first < threshold) {
          continue;
        }
        
        i32 tii = lines[i].second;
        f32 val = reductions[tii](x, y);
        f32 theta = tii/f32(reductions.size())*numPis*M_PI;
        f32 xoff = cos(theta)*length;
        f32 yoff = sin(theta)*length;
        
        pdf.setStrokeGrayLevel(val == 0.f ? 0.f :
                               val < 0.f ?
                               0.5f + val/maxAbsVal/2 :
                               0.5f + val/maxAbsVal/2);
        pdf.drawLine(x - xoff, y - yoff, x + xoff, y + yoff);
        pdf.stroke();
      }
    }
  }
  
  pdf.finish();
}

inline void WriteFlowCompatToPDF(const std::string &filename,
                                 const FlowData& flow,
                                 i32 ktToShow = -1, i32 knToShow = -1,
                                 f32 length = 0.8f,
                                 f32 threshold = 0.001f) {
  i32 flowWidth = flow[0].width();
  i32 flowHeight = flow[0].height();
  i32 ndirs = flow.size(0);
  i32 nks = flow.size(1);
  std::vector<ImageData> reduction(ndirs);
  
  f32 maxVal = -std::numeric_limits<f32>::infinity();
  for (i32 tii = 0; tii < ndirs; tii++) {
    reduction[tii] = ImageData(flowWidth, flowHeight);
    
    for (i32 ktii = 0; ktii < nks; ktii++) {
      if (ktii == ktToShow || ktToShow < 0) {
        for (i32 knii = 0; knii < nks; knii++) {
          if (knii == knToShow || knToShow < 0)
            reduction[tii].data() += flow(tii, ktii, knii).data();
        }
      }
    }
    
    maxVal = std::max(maxVal, reduction[tii].data().max());
  }
  
  std::ofstream ofs(filename.c_str());
  PDFWriter pdf(ofs, flowWidth, flowHeight);
  
  pdf.setFillGrayLevel(0.5f);
  pdf.drawRect(0, 0, flowWidth, flowHeight);
  pdf.fill();
  
  pdf.setLineCapStyle(1);
  pdf.setLineWidth(0.1f);
  pdf.translate(0.5f, 0.5f);
  
  length /= 2;
  for (i32 y = 0; y < flowHeight; y++) {
    for (i32 x = 0; x < flowWidth; x++) {
      std::vector< std::pair<f32, int> > lines(reduction.size());
      
      for (i32 tii = 0; tii < i32(lines.size()); tii++) {
        lines[tii] = std::make_pair(reduction[tii](x, y), tii);
      }
      
      std::sort(lines.begin(), lines.end());
      
      for (i32 i = 0; i < i32(lines.size()); i++) {
        f32 val = lines[i].first;
        
        if (fabs(val) < threshold) {
          continue;
        }
        
        i32 tii = lines[i].second;
        f32 theta = tii/f32(ndirs)*M_PI;
        f32 xoff = cos(theta)*length;
        f32 yoff = sin(theta)*length;
        
        pdf.setStrokeGrayLevel(0.5 + val/maxVal/2);
        pdf.drawLine(x - xoff, y - yoff, x + xoff, y + yoff);
        pdf.stroke();
      }
    }
  }
  
  pdf.finish();
}

inline void WriteFlowToPDF(const std::string &filename,
                           const FlowData& flow,
                           f32 threshold = 0.01f,
                           f32 power = 1,
                           i32 ktToShow = -1, i32 knToShow = -1,
                           f32 length = 0.8f) {
  i32 flowWidth = flow[0].width();
  i32 flowHeight = flow[0].height();
  i32 ndirs = flow.size(0);
  i32 nks = flow.size(1);
  std::vector<ImageData> reduction(ndirs);
  
  i32 nnz = 0;
  f32 maxP = 0.0f;
  f32 avgP = 0.0f;
  f32 maxVal = 0.0f;
  for (i32 tii = 0; tii < ndirs; tii++) {
    reduction[tii] = ImageData(flowWidth, flowHeight);
    
    for (i32 ktii = 0; ktii < nks; ktii++) {
      if (ktii == ktToShow || ktToShow < 0) {
        for (i32 knii = 0; knii < nks; knii++) {
          if (knii == knToShow || knToShow < 0) {
            const ImageDataValues& data = flow(tii, ktii, knii).data();
            reduction[tii].data() += data;
            maxP = std::max(maxP, data.max());
            for (i32 i = 0; i < i32(data.size()); i++) {
              if (data[i] > 0) {
                avgP += data[i];
                nnz++;
              }
            }
          }
        }
      }
    }
    
    maxVal = std::max(maxVal, reduction[tii].data().max());
  }
  
//  std::cout << "average p: " << avgP/nnz << std::endl;
//  std::cout << "max p: " << maxP << std::endl;
  
  std::ofstream ofs(filename.c_str());
  PDFWriter pdf(ofs, flowWidth, flowHeight);
 
  pdf.setLineCapStyle(1);
  pdf.setLineWidth(0.1f);
  pdf.translate(0.5f, 0.5f);
  
  length /= 2;
  for (i32 y = 0; y < flowHeight; y++) {
    for (i32 x = 0; x < flowWidth; x++) {
      std::vector<std::pair<f32, int> > lines(reduction.size());
      
      for (i32 tii = 0; tii < i32(lines.size()); tii++) {
        lines[tii] = std::make_pair(reduction[tii](x, y), tii);
      }
      
      std::sort(lines.begin(), lines.end());
      
      for (i32 i = 0; i < i32(lines.size()); i++) {
        if (lines[i].first < threshold) {
          continue;
        }
        
        i32 tii = lines[i].second;
        f32 val = powf(lines[i].first/maxVal, power);
        f32 theta = tii/f32(ndirs)*M_PI;
        f32 xoff = cos(theta)*length;
        f32 yoff = sin(theta)*length;
        
        pdf.setStrokeGrayLevel(std::max(1 - val, 0.0f));
        pdf.drawLine(x - xoff, y - yoff, x + xoff, y + yoff);
        pdf.stroke();
      }
    }
  }
  
  pdf.finish();
}

}

#endif
