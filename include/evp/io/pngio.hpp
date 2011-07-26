#pragma once
#ifndef EVP_IO_PNGIO_H
#define EVP_IO_PNGIO_H

#ifndef EVP_NO_PNG
#include <png.h>

#include <clip.hpp>

namespace evp {
using namespace clip;

inline void ReadPng(const std::string& filename, ImageData& data,
                    bool normalize = false) {
  FILE* file = fopen(filename.c_str(), "rb");
  
  if (!file)
    throw std::runtime_error("Unable to open file " + filename);
  
  u8 sig[8];
  fread(sig, 1, 8, file);
  if (!png_check_sig(sig, 8))
    throw std::invalid_argument("Image is not in png format");
  
  png_structp png_ptr =
    png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  png_infop info_ptr = png_create_info_struct(png_ptr);
  
  if (!png_ptr || !info_ptr) {
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    throw std::runtime_error("Out of memory");
  }
  
  if (setjmp(png_jmpbuf(png_ptr))) {
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    throw std::runtime_error("Error using setjmp");
  }
  
  png_init_io(png_ptr, file);
  png_set_sig_bytes(png_ptr, 8);
  png_read_info(png_ptr, info_ptr);
  
  png_uint_32 width, height;
  int bit_depth, color_type;
  png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth, &color_type,
               NULL, NULL, NULL);
  
  if (color_type == PNG_COLOR_TYPE_PALETTE ||
      (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) ||
      png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) {
    png_set_expand(png_ptr);
  }
  
  if (bit_depth == 16)
    png_set_strip_16(png_ptr);
  
  if (color_type == PNG_COLOR_TYPE_RGB ||
      color_type == PNG_COLOR_TYPE_RGB_ALPHA) {
    // Default weights for rgb -> gray, = 0.21*R + 0.72*G + 0.07*B
    png_set_rgb_to_gray(png_ptr, 1, -1, -1);
  }
  
  if (color_type & PNG_COLOR_MASK_ALPHA)
    png_set_strip_alpha(png_ptr);
  
  png_read_update_info(png_ptr, info_ptr);
  
  u32 rowbytes = png_get_rowbytes(png_ptr, info_ptr);  
  assert(rowbytes == width && "Bytes per row in PNG image not equal to width!");
  
  u8* image_data = static_cast<u8*>(malloc(rowbytes*height));
  png_bytepp row_pointers = png_bytepp(malloc(rowbytes*sizeof(png_bytep)));
  
  for (i32 i = 0; i < i32(height);  ++i)
    row_pointers[i] = image_data + (height - i - 1)*rowbytes;
  
  png_read_image(png_ptr, row_pointers);
  png_read_end(png_ptr, NULL);
  
  data = ImageData(width, height);
  
  for (i32 y = 0, i = 0; y < i32(height); ++y) {
    for (i32 x = 0; x < i32(width); ++x)
      data[i++] = *(image_data + y*rowbytes + x)/255.f;
  }
  
  if (normalize)
    data.normalize();
  
  free(row_pointers);
  free(image_data);
  png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
}

}
#endif

#endif