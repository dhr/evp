CLIP_STRINGIFY(
kernel void lland2(global imval* data1,
                   global imval* data2,
                   float degree,
                   int adapt,
                   float scale,
                   global imval* output) {
  int indx = get_global_index();
  
  float in1 = load(imval, indx, data1);
  float in2 = load(imval, indx, data2);
  
  if (adapt) {
    float maxval = fmax(in1, in2);
    degree /= maxval;
  }
  
  float cp1 = contpart(in1, degree);
  float cp2 = contpart(in2, degree);
  float prod_plus_1 = cp1*cp2 + 1;
  float result = in1*(prod_plus_1 - cp1) +
                 in2*(prod_plus_1 - cp2);
  store(imval, scale*result, indx, output);
}

kernel void llor2(global imval* data1,
                  global imval* data2,
                  float degree,
                  int adapt,
                  float scale,
                  global imval* output) {
  int indx = get_global_index();
  
  float in1 = load(imval, indx, data1);
  float in2 = load(imval, indx, data2);
  
  if (adapt) {
    float maxval = fmax(in1, in2);
    degree /= maxval;
  }
  
  float cp1 = contpart(-in1, degree);
  float cp2 = contpart(-in2, degree);
  float prod_plus_1 = cp1*cp2 + 1;
  float result = in1*(prod_plus_1 - cp1) +
                 in2*(prod_plus_1 - cp2);
  store(imval, scale*result, indx, output);
}

kernel void lland3(global imval* data1,
                   global imval* data2,
                   global imval* data3,
                   float degree,
                   int adapt,
                   float scale,
                   global imval* output) {
  int indx = get_global_index();
  
  float in1 = load(imval, indx, data1);
  float in2 = load(imval, indx, data2);
  float in3 = load(imval, indx, data3);
  
  if (adapt) {
    float maxval = fmax(in1, in2);
    maxval = fmax(maxval, in3);
    degree /= maxval;
  }
  
  float cp1 = contpart(in1, degree);
  float cp2 = contpart(in2, degree);
  float cp3 = contpart(in3, degree);
  float prod_plus_1 = cp1*cp2*cp3 + 1;
  float result = in1*(prod_plus_1 - cp1) +
                 in2*(prod_plus_1 - cp2) +
                 in3*(prod_plus_1 - cp3);
  store(imval, scale*result, indx, output);
}

kernel void llor3(global imval* data1,
                  global imval* data2,
                  global imval* data3,
                  float degree,
                  int adapt,
                  float scale,
                  global imval* output) {
  int indx = get_global_index();
  
  float in1 = load(imval, indx, data1);
  float in2 = load(imval, indx, data2);
  float in3 = load(imval, indx, data3);
  
  if (adapt) {
    float maxval = fmax(in1, in2);
    maxval = fmax(maxval, in3);
    degree /= maxval;
  }
  
  float cp1 = contpart(-in1, degree);
  float cp2 = contpart(-in2, degree);
  float cp3 = contpart(-in3, degree);
  float prod_plus_1 = cp1*cp2*cp3 + 1;
  float result = in1*(prod_plus_1 - cp1) +
                 in2*(prod_plus_1 - cp2) +
                 in3*(prod_plus_1 - cp3);
  store(imval, scale*result, indx, output);
}

kernel void lland4(global imval* data1,
                   global imval* data2,
                   global imval* data3,
                   global imval* data4,
                   float degree,
                   int adapt,
                   float scale,
                   global imval* output) {
  int indx = get_global_index();
  
  float in1 = load(imval, indx, data1);
  float in2 = load(imval, indx, data2);
  float in3 = load(imval, indx, data3);
  float in4 = load(imval, indx, data4);
  
  if (adapt) {
    float maxval = fmax(in1, in2);
    maxval = fmax(maxval, in3);
    maxval = fmax(maxval, in4);
    degree /= maxval;
  }
  
  float cp1 = contpart(in1, degree);
  float cp2 = contpart(in2, degree);
  float cp3 = contpart(in3, degree);
  float cp4 = contpart(in4, degree);
  float prod_plus_1 = cp1*cp2*cp3*cp4 + 1;
  float result = in1*(prod_plus_1 - cp1) +
                 in2*(prod_plus_1 - cp2) +
                 in3*(prod_plus_1 - cp3) +
                 in4*(prod_plus_1 - cp4);
  store(imval, scale*result, indx, output);
}

kernel void llor4(global imval* data1,
                  global imval* data2,
                  global imval* data3,
                  global imval* data4,
                  float degree,
                  int adapt,
                  float scale,
                  global imval* output) {
  int indx = get_global_index();
  
  float in1 = load(imval, indx, data1);
  float in2 = load(imval, indx, data2);
  float in3 = load(imval, indx, data3);
  float in4 = load(imval, indx, data4);
  
  if (adapt) {
    float maxval = fmax(in1, in2);
    maxval = fmax(maxval, in3);
    maxval = fmax(maxval, in4);
    degree /= maxval;
  }
  
  float cp1 = contpart(-in1, degree);
  float cp2 = contpart(-in2, degree);
  float cp3 = contpart(-in3, degree);
  float cp4 = contpart(-in4, degree);
  float prod_plus_1 = cp1*cp2*cp3*cp4 + 1;
  float result = in1*(prod_plus_1 - cp1) +
                 in2*(prod_plus_1 - cp2) +
                 in3*(prod_plus_1 - cp3) +
                 in4*(prod_plus_1 - cp4);
  store(imval, scale*result, indx, output);
}

kernel void lland5(global imval* data1,
                   global imval* data2,
                   global imval* data3,
                   global imval* data4,
                   global imval* data5,
                   float degree,
                   int adapt,
                   float scale,
                   global imval* output) {
  int indx = get_global_index();
  
  float in1 = load(imval, indx, data1);
  float in2 = load(imval, indx, data2);
  float in3 = load(imval, indx, data3);
  float in4 = load(imval, indx, data4);
  float in5 = load(imval, indx, data5);
  
  if (adapt) {
    float maxval = fmax(in1, in2);
    maxval = fmax(maxval, in3);
    maxval = fmax(maxval, in4);
    maxval = fmax(maxval, in5);
    degree /= maxval;
  }
  
  float cp1 = contpart(in1, degree);
  float cp2 = contpart(in2, degree);
  float cp3 = contpart(in3, degree);
  float cp4 = contpart(in4, degree);
  float cp5 = contpart(in5, degree);
  float prod_plus_1 = cp1*cp2*cp3*cp4*cp5 + 1;
  float result = in1*(prod_plus_1 - cp1) +
                 in2*(prod_plus_1 - cp2) +
                 in3*(prod_plus_1 - cp3) +
                 in4*(prod_plus_1 - cp4) +
                 in5*(prod_plus_1 - cp5);
  store(imval, scale*result, indx, output);
}

kernel void llor5(global imval* data1,
                  global imval* data2,
                  global imval* data3,
                  global imval* data4,
                  global imval* data5,
                  float degree,
                  int adapt,
                  float scale,
                  global imval* output) {
  int indx = get_global_index();
  
  float in1 = load(imval, indx, data1);
  float in2 = load(imval, indx, data2);
  float in3 = load(imval, indx, data3);
  float in4 = load(imval, indx, data4);
  float in5 = load(imval, indx, data5);
  
  if (adapt) {
    float maxval = fmax(in1, in2);
    maxval = fmax(maxval, in3);
    maxval = fmax(maxval, in4);
    maxval = fmax(maxval, in5);
    degree /= maxval;
  }
  
  float cp1 = contpart(-in1, degree);
  float cp2 = contpart(-in2, degree);
  float cp3 = contpart(-in3, degree);
  float cp4 = contpart(-in4, degree);
  float cp5 = contpart(-in5, degree);
  float prod_plus_1 = cp1*cp2*cp3*cp4*cp5 + 1;
  float result = in1*(prod_plus_1 - cp1) +
                 in2*(prod_plus_1 - cp2) +
                 in3*(prod_plus_1 - cp3) +
                 in4*(prod_plus_1 - cp4) +
                 in5*(prod_plus_1 - cp5);
  store(imval, scale*result, indx, output);
}

kernel void lland6(global imval* data1,
                   global imval* data2,
                   global imval* data3,
                   global imval* data4,
                   global imval* data5,
                   global imval* data6,
                   float degree,
                   int adapt,
                   float scale,
                   global imval* output) {
  int indx = get_global_index();
  
  float in1 = load(imval, indx, data1);
  float in2 = load(imval, indx, data2);
  float in3 = load(imval, indx, data3);
  float in4 = load(imval, indx, data4);
  float in5 = load(imval, indx, data5);
  float in6 = load(imval, indx, data6);
  
  if (adapt) {
    float maxval = fmax(in1, in2);
    maxval = fmax(maxval, in3);
    maxval = fmax(maxval, in4);
    maxval = fmax(maxval, in5);
    maxval = fmax(maxval, in6);
    degree /= maxval;
  }
  
  float cp1 = contpart(in1, degree);
  float cp2 = contpart(in2, degree);
  float cp3 = contpart(in3, degree);
  float cp4 = contpart(in4, degree);
  float cp5 = contpart(in5, degree);
  float cp6 = contpart(in6, degree);
  float prod_plus_1 = cp1*cp2*cp3*cp4*cp5*cp6 + 1;
  float result = in1*(prod_plus_1 - cp1) +
                 in2*(prod_plus_1 - cp2) +
                 in3*(prod_plus_1 - cp3) +
                 in4*(prod_plus_1 - cp4) +
                 in5*(prod_plus_1 - cp5) +
                 in6*(prod_plus_1 - cp6);
  store(imval, scale*result, indx, output);
}

kernel void llor6(global imval* data1,
                  global imval* data2,
                  global imval* data3,
                  global imval* data4,
                  global imval* data5,
                  global imval* data6,
                  float degree,
                  int adapt,
                  float scale,
                  global imval* output) {
  int indx = get_global_index();
  
  float in1 = load(imval, indx, data1);
  float in2 = load(imval, indx, data2);
  float in3 = load(imval, indx, data3);
  float in4 = load(imval, indx, data4);
  float in5 = load(imval, indx, data5);
  float in6 = load(imval, indx, data6);
  
  if (adapt) {
    float maxval = fmax(in1, in2);
    maxval = fmax(maxval, in3);
    maxval = fmax(maxval, in4);
    maxval = fmax(maxval, in5);
    maxval = fmax(maxval, in6);
    degree /= maxval;
  }
  
  float cp1 = contpart(-in1, degree);
  float cp2 = contpart(-in2, degree);
  float cp3 = contpart(-in3, degree);
  float cp4 = contpart(-in4, degree);
  float cp5 = contpart(-in5, degree);
  float cp6 = contpart(-in6, degree);
  float prod_plus_1 = cp1*cp2*cp3*cp4*cp5*cp6 + 1;
  float result = in1*(prod_plus_1 - cp1) +
                 in2*(prod_plus_1 - cp2) +
                 in3*(prod_plus_1 - cp3) +
                 in4*(prod_plus_1 - cp4) +
                 in5*(prod_plus_1 - cp5) +
                 in6*(prod_plus_1 - cp6);
  store(imval, scale*result, indx, output);
}
)
