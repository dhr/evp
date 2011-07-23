#pragma once
#ifndef EVP_IO_IMAGEIO_H
#define EVP_IO_IMAGEIO_H

#include <cstdio>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <valarray>

#ifndef EVP_NO_JPEG
#include "evp/io/jpegio.hpp"
#endif

#ifndef EVP_NO_PNG
#include "evp/io/pngio.hpp"
#endif

#include <clip.hpp>

namespace evp {
using namespace clip;

inline void WriteCSV(const std::string &filename, const ImageData &data,
                     i32 precision = 10, bool sciNot = true) {
  std::ofstream ofs(filename.c_str());
  
  if (sciNot) {
    ofs << std::scientific;
  } else {
    ofs << std::fixed;
  }
  
  ofs << std::setprecision(precision);
  i32 width = data.width();
  i32 height = data.height();
  for (i32 y = height - 1; y < height; --y) {
    for (i32 x = 0; x < width - 1; ++x)
      ofs << std::setw(precision + 3) << data(x, y) << ", ";
    ofs << std::setw(precision + 3)
    << data(data.width() - 1, y) << std::endl;
  }
}

inline void WriteCSV(const char *filename, f32 *data, i32 w, i32 h,
                     i32 precision = 10, bool sciNot = true) {
  std::ofstream ofs(filename);
  
  if (sciNot) {
    ofs << std::scientific;
  } else {
    ofs << std::fixed;
  }
  
  ofs << std::setprecision(precision);
  for (i32 y = h - 1; y >= 0; --y) {
    for (i32 x = 0; x < w - 1; ++x)
      ofs << std::setw(precision + 3) << data[y*w + x] << ", ";
    ofs << std::setw(precision + 3) << data[y*w + w - 1] << std::endl;
  }
}

inline void OutputStack(ImBufList& list, bool writeCSV = false) {
  ImBufList::iterator it = list.begin();
  for (i32 i = 0; it != list.end(); it++, i++) {
    ImageData imData = it->fetchData();
    
    std::stringstream name;
    name << "Output/dbg-im-1-" << std::setfill('0') << std::setw(2)
         << i << ".jpg";
    WriteJpeg(name.str().c_str(), imData, true);
    
    if (writeCSV) {
      name.str("");
      name << "Output/csv-im-1-" << std::setfill('0') << std::setw(2)
           << i << ".csv";
      WriteCSV(name.str().c_str(), imData);
    }
  }
}

inline void OutputBuffer(const ImageBuffer& buf, bool normalize = true) {
  WriteJpeg("Output/buffer.jpg", buf.fetchData(), normalize);
}

}

#endif
