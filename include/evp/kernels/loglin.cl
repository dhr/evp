CLIP_STRINGIFY(
kernel void lland2(global float *data1,
                   global float *data2,
                   float degree,
                   int adapt,
                   float scale,
                   global float *output) {
  int indx = get_global_index();
  
  if (adapt) {
    float maxval = fmax(data1[indx], data2[indx]);
    degree /= maxval;
  }
  
  float cp1 = contpart(data1[indx], degree);
  float cp2 = contpart(data2[indx], degree);
  float prod_plus_1 = cp1*cp2 + 1;
  float result = data1[indx]*(prod_plus_1 - cp1) +
  data2[indx]*(prod_plus_1 - cp2);
  output[indx] = scale*result;
}

kernel void llor2(global float *data1,
                  global float *data2,
                  float degree,
                  int adapt,
                  float scale,
                  global float *output) {
  int indx = get_global_index();
  
  if (adapt) {
    float maxval = fmax(data1[indx], data2[indx]);
    degree /= maxval;
  }
  
  float cp1 = contpart(-data1[indx], degree);
  float cp2 = contpart(-data2[indx], degree);
  float prod_plus_1 = cp1*cp2 + 1;
  float result = data1[indx]*(prod_plus_1 - cp1) +
  data2[indx]*(prod_plus_1 - cp2);
  output[indx] = scale*result;
}

kernel void lland3(global float *data1,
                   global float *data2,
                   global float *data3,
                   float degree,
                   int adapt,
                   float scale,
                   global float *output) {
  int indx = get_global_index();
  
  if (adapt) {
    float maxval = fmax(data1[indx], data2[indx]);
    maxval = fmax(maxval, data3[indx]);
    degree /= maxval;
  }
  
  float cp1 = contpart(data1[indx], degree);
  float cp2 = contpart(data2[indx], degree);
  float cp3 = contpart(data3[indx], degree);
  float prod_plus_1 = cp1*cp2*cp3 + 1;
  float result = data1[indx]*(prod_plus_1 - cp1) +
  data2[indx]*(prod_plus_1 - cp2) +
  data3[indx]*(prod_plus_1 - cp3);
  output[indx] = scale*result;
}

kernel void llor3(global float *data1,
                  global float *data2,
                  global float *data3,
                  float degree,
                  int adapt,
                  float scale,
                  global float *output) {
  int indx = get_global_index();
  
  if (adapt) {
    float maxval = fmax(data1[indx], data2[indx]);
    maxval = fmax(maxval, data3[indx]);
    degree /= maxval;
  }
  
  float cp1 = contpart(-data1[indx], degree);
  float cp2 = contpart(-data2[indx], degree);
  float cp3 = contpart(-data3[indx], degree);
  float prod_plus_1 = cp1*cp2*cp3 + 1;
  float result = data1[indx]*(prod_plus_1 - cp1) +
  data2[indx]*(prod_plus_1 - cp2) +
  data3[indx]*(prod_plus_1 - cp3);
  output[indx] = scale*result;
}

kernel void lland4(global float *data1,
                   global float *data2,
                   global float *data3,
                   global float *data4,
                   float degree,
                   int adapt,
                   float scale,
                   global float *output) {
  int indx = get_global_index();
  
  if (adapt) {
    float maxval = fmax(data1[indx], data2[indx]);
    maxval = fmax(maxval, data3[indx]);
    maxval = fmax(maxval, data4[indx]);
    degree /= maxval;
  }
  
  float cp1 = contpart(data1[indx], degree);
  float cp2 = contpart(data2[indx], degree);
  float cp3 = contpart(data3[indx], degree);
  float cp4 = contpart(data4[indx], degree);
  float prod_plus_1 = cp1*cp2*cp3*cp4 + 1;
  float result = data1[indx]*(prod_plus_1 - cp1) +
  data2[indx]*(prod_plus_1 - cp2) +
  data3[indx]*(prod_plus_1 - cp3) +
  data4[indx]*(prod_plus_1 - cp4);
  output[indx] = scale*result;
}

kernel void llor4(global float *data1,
                  global float *data2,
                  global float *data3,
                  global float *data4,
                  float degree,
                  int adapt,
                  float scale,
                  global float *output) {
  int indx = get_global_index();
  
  if (adapt) {
    float maxval = fmax(data1[indx], data2[indx]);
    maxval = fmax(maxval, data3[indx]);
    maxval = fmax(maxval, data4[indx]);
    degree /= maxval;
  }
  
  float cp1 = contpart(-data1[indx], degree);
  float cp2 = contpart(-data2[indx], degree);
  float cp3 = contpart(-data3[indx], degree);
  float cp4 = contpart(-data4[indx], degree);
  float prod_plus_1 = cp1*cp2*cp3*cp4 + 1;
  float result = data1[indx]*(prod_plus_1 - cp1) +
  data2[indx]*(prod_plus_1 - cp2) +
  data3[indx]*(prod_plus_1 - cp3) +
  data4[indx]*(prod_plus_1 - cp4);
  output[indx] = scale*result;
}

kernel void lland5(global float *data1,
                   global float *data2,
                   global float *data3,
                   global float *data4,
                   global float *data5,
                   float degree,
                   int adapt,
                   float scale,
                   global float *output) {
  int indx = get_global_index();
  
  if (adapt) {
    float maxval = fmax(data1[indx], data2[indx]);
    maxval = fmax(maxval, data3[indx]);
    maxval = fmax(maxval, data4[indx]);
    maxval = fmax(maxval, data5[indx]);
    degree /= maxval;
  }
  
  float cp1 = contpart(data1[indx], degree);
  float cp2 = contpart(data2[indx], degree);
  float cp3 = contpart(data3[indx], degree);
  float cp4 = contpart(data4[indx], degree);
  float cp5 = contpart(data5[indx], degree);
  float prod_plus_1 = cp1*cp2*cp3*cp4*cp5 + 1;
  float result = data1[indx]*(prod_plus_1 - cp1) +
  data2[indx]*(prod_plus_1 - cp2) +
  data3[indx]*(prod_plus_1 - cp3) +
  data4[indx]*(prod_plus_1 - cp4) +
  data5[indx]*(prod_plus_1 - cp5);
  output[indx] = scale*result;
}

kernel void llor5(global float *data1,
                  global float *data2,
                  global float *data3,
                  global float *data4,
                  global float *data5,
                  float degree,
                  int adapt,
                  float scale,
                  global float *output) {
  int indx = get_global_index();
  
  if (adapt) {
    float maxval = fmax(data1[indx], data2[indx]);
    maxval = fmax(maxval, data3[indx]);
    maxval = fmax(maxval, data4[indx]);
    maxval = fmax(maxval, data5[indx]);
    degree /= maxval;
  }
  
  float cp1 = contpart(-data1[indx], degree);
  float cp2 = contpart(-data2[indx], degree);
  float cp3 = contpart(-data3[indx], degree);
  float cp4 = contpart(-data4[indx], degree);
  float cp5 = contpart(-data5[indx], degree);
  float prod_plus_1 = cp1*cp2*cp3*cp4*cp5 + 1;
  float result = data1[indx]*(prod_plus_1 - cp1) +
  data2[indx]*(prod_plus_1 - cp2) +
  data3[indx]*(prod_plus_1 - cp3) +
  data4[indx]*(prod_plus_1 - cp4) +
  data5[indx]*(prod_plus_1 - cp5);
  output[indx] = scale*result;
}

kernel void lland6(global float *data1,
                   global float *data2,
                   global float *data3,
                   global float *data4,
                   global float *data5,
                   global float *data6,
                   float degree,
                   int adapt,
                   float scale,
                   global float *output) {
  int indx = get_global_index();
  
  if (adapt) {
    float maxval = fmax(data1[indx], data2[indx]);
    maxval = fmax(maxval, data3[indx]);
    maxval = fmax(maxval, data4[indx]);
    maxval = fmax(maxval, data5[indx]);
    maxval = fmax(maxval, data6[indx]);
    degree /= maxval;
  }
  
  float cp1 = contpart(data1[indx], degree);
  float cp2 = contpart(data2[indx], degree);
  float cp3 = contpart(data3[indx], degree);
  float cp4 = contpart(data4[indx], degree);
  float cp5 = contpart(data5[indx], degree);
  float cp6 = contpart(data6[indx], degree);
  float prod_plus_1 = cp1*cp2*cp3*cp4*cp5*cp6 + 1;
  float result = data1[indx]*(prod_plus_1 - cp1) +
  data2[indx]*(prod_plus_1 - cp2) +
  data3[indx]*(prod_plus_1 - cp3) +
  data4[indx]*(prod_plus_1 - cp4) +
  data5[indx]*(prod_plus_1 - cp5) +
  data6[indx]*(prod_plus_1 - cp6);
  output[indx] = scale*result;
}

kernel void llor6(global float *data1,
                  global float *data2,
                  global float *data3,
                  global float *data4,
                  global float *data5,
                  global float *data6,
                  float degree,
                  int adapt,
                  float scale,
                  global float *output) {
  int indx = get_global_index();
  
  if (adapt) {
    float maxval = fmax(data1[indx], data2[indx]);
    maxval = fmax(maxval, data3[indx]);
    maxval = fmax(maxval, data4[indx]);
    maxval = fmax(maxval, data5[indx]);
    maxval = fmax(maxval, data6[indx]);
    degree /= maxval;
  }
  
  float cp1 = contpart(-data1[indx], degree);
  float cp2 = contpart(-data2[indx], degree);
  float cp3 = contpart(-data3[indx], degree);
  float cp4 = contpart(-data4[indx], degree);
  float cp5 = contpart(-data5[indx], degree);
  float cp6 = contpart(-data6[indx], degree);
  float prod_plus_1 = cp1*cp2*cp3*cp4*cp5*cp6 + 1;
  float result = data1[indx]*(prod_plus_1 - cp1) +
  data2[indx]*(prod_plus_1 - cp2) +
  data3[indx]*(prod_plus_1 - cp3) +
  data4[indx]*(prod_plus_1 - cp4) +
  data5[indx]*(prod_plus_1 - cp5) +
  data6[indx]*(prod_plus_1 - cp6);
  output[indx] = scale*result;
}
)
