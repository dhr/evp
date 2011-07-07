#pragma once
#ifndef EVP_IO_IMAGEIO_H
#define EVP_IO_IMAGEIO_H

#include <cstdio>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <valarray>

#include <jpeglib.h>

#include <clip.hpp>

namespace evp {
using namespace clip;

inline bool ReadJpeg(const std::string &filename, ImageData &data,
                     bool normalize = false) {
  FILE *infile;
  if ((infile = fopen(filename.c_str(), "rb")) == NULL)
    return false;
  
  struct jpeg_decompress_struct dinfo;
  struct jpeg_error_mgr jerr;
  
  dinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&dinfo);
  
  jpeg_stdio_src(&dinfo, infile);
  jpeg_read_header(&dinfo, TRUE);
  jpeg_start_decompress(&dinfo);
  
  i32 w = dinfo.output_width;
  i32 h = dinfo.output_height;
  i32 nelems = w*h;
  
  u8 *imdata = (u8*) malloc(nelems*dinfo.output_components);
  while (dinfo.output_scanline < dinfo.image_height) {
    i32 offset = dinfo.output_width*dinfo.output_components;
    i32 scanline = dinfo.output_scanline;
    u8 *row_data = &imdata[offset*(h - 1 - scanline)];
    jpeg_read_scanlines(&dinfo, &row_data, 1);
  }
  i32 components = dinfo.output_components;
  
  jpeg_finish_decompress(&dinfo);
  jpeg_destroy_decompress(&dinfo);
  fclose(infile);
  
  data = ImageData(w, h);
  
  std::valarray<f32> &gray = data.data();
  for (i32 i = 0, c = 0; i < nelems; i++, c += components) {
    f32 sum = 0;
    for (i32 j = 0; j < components; j++)
      sum += imdata[c + j];
    gray[i] = sum/components/256;
  }
  
  if (normalize) {
    gray /= gray.max();
  }
  
  free(imdata);
  
  return true;
}

inline bool WriteJpeg(const std::string &filename, const ImageData &data,
                      bool normalize = false) {
  FILE *outfile;
	if ((outfile = fopen(filename.c_str(), "wb")) == NULL)
    return false;
  
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  
  cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_compress(&cinfo);
	jpeg_stdio_dest(&cinfo, outfile);
  
	cinfo.image_width = data.width();	
	cinfo.image_height = data.height();
	cinfo.input_components = 1;
	cinfo.in_color_space = JCS_GRAYSCALE;
  
	jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, 100, TRUE);
	jpeg_start_compress(&cinfo, TRUE);
  
  f32 min = data.data().min();
  f32 max = data.data().max();
  u8 *imdata = (u8*) (cinfo.image_width);
	while (cinfo.next_scanline < cinfo.image_height) {
    for (JDIMENSION c = 0; c < cinfo.image_width; c++) {
      f32 val = data(c, cinfo.image_height - 1 - cinfo.next_scanline);
      if (normalize) val = (val - min)/(max - min);
      imdata[c] = (u8) (val*255);
    }
		jpeg_write_scanlines(&cinfo, &imdata, 1);
	}
  
	jpeg_finish_compress(&cinfo);
	jpeg_destroy_compress(&cinfo);
	fclose(outfile);
  
  return true;
}

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
