#pragma once
#ifndef EVP_IO_PDFWRITER_H
#define EVP_IO_PDFWRITER_H

#include <fstream>
#include <vector>

namespace evp {

class PDFWriter {
  std::ofstream &os_;
  int width_;
  int height_;
  std::vector<int> offsets_;
  int streamStart_;
  
  void writePreamble() {
    os_ <<
    "%PDF-1.4" << std::endl;
    
    offsets_.push_back(int(os_.tellp()));
    os_ <<
    "1 0 obj" << std::endl <<
    "<<" << std::endl <<
    "/Type /Catalog" << std::endl <<
    "/Outlines 2 0 R" << std::endl <<
    "/Pages 3 0 R" << std::endl <<
    ">>" << std::endl <<
    "endobj" << std::endl << std::endl;
    
    offsets_.push_back(int(os_.tellp()));
    os_ <<
    "2 0 obj" << std::endl <<
    "<<" << std::endl <<
    "/Type /Outlines" << std::endl <<
    "/Count 0" << std::endl <<
    ">>" << std::endl <<
    "endobj" << std::endl << std::endl;
    
    offsets_.push_back(int(os_.tellp()));
    os_ <<
    "3 0 obj" << std::endl <<
    "<<" << std::endl <<
    "/Type /Pages" << std::endl <<
    "/Kids [4 0 R]" << std::endl <<
    "/Count 1" << std::endl <<
    "/MediaBox [0 0 " << width_ << ' ' << height_ << ']' << std::endl <<
    ">>" << std::endl <<
    "endobj" << std::endl << std::endl;
    
    offsets_.push_back(int(os_.tellp()));
    os_ <<
    "4 0 obj" << std::endl <<
    "<<" << std::endl <<
    "/Type /Page" << std::endl <<
    "/Parent 3 0 R" << std::endl <<
    "/Contents 6 0 R" << std::endl <<
    "/Resources <</ProcSet 5 0 R>>" << std::endl <<
    ">>" << std::endl <<
    "endobj" << std::endl << std::endl;
    
    offsets_.push_back(int(os_.tellp()));
    os_ <<
    "5 0 obj" << std::endl <<
    "[/PDF]" << std::endl <<
    "endobj" << std::endl << std::endl;
    
    offsets_.push_back(int(os_.tellp()));
    os_ <<
    "6 0 obj" << std::endl <<
    "<</Length 7 0 R>>" << std::endl <<
    "stream" << std::endl;
    
    streamStart_ = int(os_.tellp());
  }
  
 public:
  PDFWriter(std::ofstream &outputStream, int width, int height)
  : os_(outputStream), width_(width), height_(height)
  {
    writePreamble();
    os_ << std::fixed << std::setprecision(3);
  }
  
  void setStrokeGrayLevel(float gl) { os_ << gl << " G" << std::endl; }
  
  void setFillGrayLevel(float gl) { os_ << gl << " g" << std::endl; }
  
  void setLineCapStyle(int lcs) { os_ << lcs << " J" << std::endl; }
  
  void setLineWidth(float lw) { os_ << lw << " w" << std::endl; }
  
  void translate(float tx, float ty) {
    os_ << "1 0 0 1 " << tx << ' ' << ty << " cm" << std::endl;
  }
  
  void moveTo(float x, float y) {
    os_ << x << ' ' << y << " m" << std::endl;
  }
  
  void lineTo(float x, float y) {
    os_ << x << ' ' << y << " l" << std::endl;
  }
  
  void curveTo(float x1, float y1, float x2, float y2, float x3, float y3) {
    os_ << x1 << ' ' << y1 << ' '
    << x2 << ' ' << y2 << ' '
    << x3 << ' ' << y3 << " c" << std::endl;
  }
  
  void closePath() {
    os_ << "h" << std::endl;
  }
  
  void drawRect(float x, float y, float w, float h) {
    os_ << x << ' ' << y << ' ' << w << ' ' << h << " re" << std::endl;
  }
  
  void drawArc(float centerX, float centerY, float radius,
               float startTheta, float endTheta) {
    float cos1 = cos(startTheta), sin1 = sin(startTheta);
    float cos2 = cos(endTheta), sin2 = sin(endTheta);
    float p1[2] = {centerX + radius*cos1, centerY + radius*sin1};
    float p2[2] = {centerX + radius*cos2, centerY + radius*sin2};
    
    float diffTheta = (endTheta - startTheta)/2;
    float ctrlLen = (4.f/3.f)*radius*(1 - cos(diffTheta))/sin(diffTheta);
    float c1[2] = {p1[0] - ctrlLen*sin1, p1[1] + ctrlLen*cos1};
    float c2[2] = {p2[0] + ctrlLen*sin2, p2[1] - ctrlLen*cos2};
    
    moveTo(p1[0], p1[1]);
    curveTo(c1[0], c1[1], c2[0], c2[1], p2[0], p2[1]);
  }
  
  void drawLine(float fromX, float fromY, float toX, float toY) {
    moveTo(fromX, fromY);
    lineTo(toX, toY);
  }
  
  void stroke() { os_ << "S" << std::endl; }
  
  void fill() { os_ << "f" << std::endl; }
  
  void finish() {
    int streamEnd = int(os_.tellp());
#if _WIN32 // Not the right way to do this, but I'm lazy
    bool extraSpace = false;
#else
    bool extraSpace = true;
#endif
    
    os_ <<
    "endstream" << std::endl <<
    "endobj" << std::endl << std::endl;
    
    offsets_.push_back(int(os_.tellp()));
    os_ <<
    "7 0 obj" << std::endl <<
    streamEnd - streamStart_ << std::endl <<
    "endobj" << std::endl << std::endl;
    
    int xrefOffset = int(os_.tellp());
    os_ <<
    "xref" << std::endl <<
    "0 " << offsets_.size() + 1 << std::endl <<
    "0000000000 65535 f" << (extraSpace ? " " : "") << std::endl;
    
    os_ << std::setfill('0');
    std::vector<int>::iterator it, end;
    for (it = offsets_.begin(), end = offsets_.end(); it != end; ++it) {
      os_ << std::setw(10) << *it;
      os_ << std::setw(0) << " 00000 n";
      os_ << (extraSpace ? " " : "") << std::endl;
    }
    
    os_ << std::endl <<
    "trailer" << std::endl <<
    "<<" << std::endl <<
    "/Size " << offsets_.size() + 1 << std::endl <<
    "/Root 1 0 R" << std::endl <<
    ">>" << std::endl <<
    "startxref" << std::endl <<
    xrefOffset << std::endl <<
    "%%EOF" << std::endl;
    os_.close();
  }
  
};

}

#endif
