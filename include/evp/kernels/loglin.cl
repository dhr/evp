kernel void lland2(input_t data1,
                   input_t data2,
                   float degree_val,
                   char adapt,
                   float scale,
                   output_t output) {
  index_t indx = get_global_index();
  calc_t degree = (calc_t) degree_val;
  
  calc_t in1 = load(indx, data1);
  calc_t in2 = load(indx, data2);
  
  if (adapt) {
    calc_t maxval = fmax(in1, in2);
    degree /= maxval;
  }
  
  calc_t cp1 = contpart(in1, degree);
  calc_t cp2 = contpart(in2, degree);
  calc_t prod_plus_1 = cp1*cp2 + 1.f;
  calc_t result = in1*(prod_plus_1 - cp1) +
                  in2*(prod_plus_1 - cp2);
  store(scale*result, indx, output);
}

kernel void llor2(input_t data1,
                  input_t data2,
                  float degree_val,
                  char adapt,
                  float scale,
                  output_t output) {
  index_t indx = get_global_index();
  calc_t degree = (calc_t) degree_val;
  
  calc_t in1 = load(indx, data1);
  calc_t in2 = load(indx, data2);
  
  if (adapt) {
    calc_t maxval = fmax(in1, in2);
    degree /= maxval;
  }
  
  calc_t cp1 = contpart(-in1, degree);
  calc_t cp2 = contpart(-in2, degree);
  calc_t prod_plus_1 = cp1*cp2 + 1.f;
  calc_t result = in1*(prod_plus_1 - cp1) +
                  in2*(prod_plus_1 - cp2);
  store(scale*result, indx, output);
}

kernel void lland3(input_t data1,
                   input_t data2,
                   input_t data3,
                   float degree_val,
                   char adapt,
                   float scale,
                   output_t output) {
  index_t indx = get_global_index();
  calc_t degree = (calc_t) degree_val;
  
  calc_t in1 = load(indx, data1);
  calc_t in2 = load(indx, data2);
  calc_t in3 = load(indx, data3);
  
  if (adapt) {
    calc_t maxval = fmax(in1, in2);
    maxval = fmax(maxval, in3);
    degree /= maxval;
  }
  
  calc_t cp1 = contpart(in1, degree);
  calc_t cp2 = contpart(in2, degree);
  calc_t cp3 = contpart(in3, degree);
  calc_t prod_plus_1 = cp1*cp2*cp3 + 1.f;
  calc_t result = in1*(prod_plus_1 - cp1) +
                  in2*(prod_plus_1 - cp2) +
                  in3*(prod_plus_1 - cp3);
  store(scale*result, indx, output);
}

kernel void llor3(input_t data1,
                  input_t data2,
                  input_t data3,
                  float degree_val,
                  char adapt,
                  float scale,
                  output_t output) {
  index_t indx = get_global_index();
  calc_t degree = (calc_t) degree_val;
  
  calc_t in1 = load(indx, data1);
  calc_t in2 = load(indx, data2);
  calc_t in3 = load(indx, data3);
  
  if (adapt) {
    calc_t maxval = fmax(in1, in2);
    maxval = fmax(maxval, in3);
    degree /= maxval;
  }
  
  calc_t cp1 = contpart(-in1, degree);
  calc_t cp2 = contpart(-in2, degree);
  calc_t cp3 = contpart(-in3, degree);
  calc_t prod_plus_1 = cp1*cp2*cp3 + 1.f;
  calc_t result = in1*(prod_plus_1 - cp1) +
                  in2*(prod_plus_1 - cp2) +
                  in3*(prod_plus_1 - cp3);
  store(scale*result, indx, output);
}

kernel void lland4(input_t data1,
                   input_t data2,
                   input_t data3,
                   input_t data4,
                   float degree_val,
                   char adapt,
                   float scale,
                   output_t output) {
  index_t indx = get_global_index();
  calc_t degree = (calc_t) degree_val;
  
  calc_t in1 = load(indx, data1);
  calc_t in2 = load(indx, data2);
  calc_t in3 = load(indx, data3);
  calc_t in4 = load(indx, data4);
  
  if (adapt) {
    calc_t maxval = fmax(in1, in2);
    maxval = fmax(maxval, in3);
    maxval = fmax(maxval, in4);
    degree /= maxval;
  }
  
  calc_t cp1 = contpart(in1, degree);
  calc_t cp2 = contpart(in2, degree);
  calc_t cp3 = contpart(in3, degree);
  calc_t cp4 = contpart(in4, degree);
  calc_t prod_plus_1 = cp1*cp2*cp3*cp4 + 1.f;
  calc_t result = in1*(prod_plus_1 - cp1) +
                  in2*(prod_plus_1 - cp2) +
                  in3*(prod_plus_1 - cp3) +
                  in4*(prod_plus_1 - cp4);
  store(scale*result, indx, output);
}

kernel void llor4(input_t data1,
                  input_t data2,
                  input_t data3,
                  input_t data4,
                  float degree_val,
                  char adapt,
                  float scale,
                  output_t output) {
  index_t indx = get_global_index();
  calc_t degree = (calc_t) degree_val;
  
  calc_t in1 = load(indx, data1);
  calc_t in2 = load(indx, data2);
  calc_t in3 = load(indx, data3);
  calc_t in4 = load(indx, data4);
  
  if (adapt) {
    calc_t maxval = fmax(in1, in2);
    maxval = fmax(maxval, in3);
    maxval = fmax(maxval, in4);
    degree /= maxval;
  }
  
  calc_t cp1 = contpart(-in1, degree);
  calc_t cp2 = contpart(-in2, degree);
  calc_t cp3 = contpart(-in3, degree);
  calc_t cp4 = contpart(-in4, degree);
  calc_t prod_plus_1 = cp1*cp2*cp3*cp4 + 1.f;
  calc_t result = in1*(prod_plus_1 - cp1) +
                  in2*(prod_plus_1 - cp2) +
                  in3*(prod_plus_1 - cp3) +
                  in4*(prod_plus_1 - cp4);
  store(scale*result, indx, output);
}

kernel void lland5(input_t data1,
                   input_t data2,
                   input_t data3,
                   input_t data4,
                   input_t data5,
                   float degree_val,
                   char adapt,
                   float scale,
                   output_t output) {
  index_t indx = get_global_index();
  calc_t degree = (calc_t) degree_val;
  
  calc_t in1 = load(indx, data1);
  calc_t in2 = load(indx, data2);
  calc_t in3 = load(indx, data3);
  calc_t in4 = load(indx, data4);
  calc_t in5 = load(indx, data5);
  
  if (adapt) {
    calc_t maxval = fmax(in1, in2);
    maxval = fmax(maxval, in3);
    maxval = fmax(maxval, in4);
    maxval = fmax(maxval, in5);
    degree /= maxval;
  }
  
  calc_t cp1 = contpart(in1, degree);
  calc_t cp2 = contpart(in2, degree);
  calc_t cp3 = contpart(in3, degree);
  calc_t cp4 = contpart(in4, degree);
  calc_t cp5 = contpart(in5, degree);
  calc_t prod_plus_1 = cp1*cp2*cp3*cp4*cp5 + 1.f;
  calc_t result = in1*(prod_plus_1 - cp1) +
                  in2*(prod_plus_1 - cp2) +
                  in3*(prod_plus_1 - cp3) +
                  in4*(prod_plus_1 - cp4) +
                  in5*(prod_plus_1 - cp5);
  store(scale*result, indx, output);
}

kernel void llor5(input_t data1,
                  input_t data2,
                  input_t data3,
                  input_t data4,
                  input_t data5,
                  float degree_val,
                  char adapt,
                  float scale,
                  output_t output) {
  index_t indx = get_global_index();
  calc_t degree = (calc_t) degree_val;
  
  calc_t in1 = load(indx, data1);
  calc_t in2 = load(indx, data2);
  calc_t in3 = load(indx, data3);
  calc_t in4 = load(indx, data4);
  calc_t in5 = load(indx, data5);
  
  if (adapt) {
    calc_t maxval = fmax(in1, in2);
    maxval = fmax(maxval, in3);
    maxval = fmax(maxval, in4);
    maxval = fmax(maxval, in5);
    degree /= maxval;
  }
  
  calc_t cp1 = contpart(-in1, degree);
  calc_t cp2 = contpart(-in2, degree);
  calc_t cp3 = contpart(-in3, degree);
  calc_t cp4 = contpart(-in4, degree);
  calc_t cp5 = contpart(-in5, degree);
  calc_t prod_plus_1 = cp1*cp2*cp3*cp4*cp5 + 1.f;
  calc_t result = in1*(prod_plus_1 - cp1) +
                  in2*(prod_plus_1 - cp2) +
                  in3*(prod_plus_1 - cp3) +
                  in4*(prod_plus_1 - cp4) +
                  in5*(prod_plus_1 - cp5);
  store(scale*result, indx, output);
}

kernel void lland6(input_t data1,
                   input_t data2,
                   input_t data3,
                   input_t data4,
                   input_t data5,
                   input_t data6,
                   float degree_val,
                   char adapt,
                   float scale,
                   output_t output) {
  index_t indx = get_global_index();
  calc_t degree = (calc_t) degree_val;
  
  calc_t in1 = load(indx, data1);
  calc_t in2 = load(indx, data2);
  calc_t in3 = load(indx, data3);
  calc_t in4 = load(indx, data4);
  calc_t in5 = load(indx, data5);
  calc_t in6 = load(indx, data6);
  
  if (adapt) {
    calc_t maxval = fmax(in1, in2);
    maxval = fmax(maxval, in3);
    maxval = fmax(maxval, in4);
    maxval = fmax(maxval, in5);
    maxval = fmax(maxval, in6);
    degree /= maxval;
  }
  
  calc_t cp1 = contpart(in1, degree);
  calc_t cp2 = contpart(in2, degree);
  calc_t cp3 = contpart(in3, degree);
  calc_t cp4 = contpart(in4, degree);
  calc_t cp5 = contpart(in5, degree);
  calc_t cp6 = contpart(in6, degree);
  calc_t prod_plus_1 = cp1*cp2*cp3*cp4*cp5*cp6 + 1.f;
  calc_t result = in1*(prod_plus_1 - cp1) +
                  in2*(prod_plus_1 - cp2) +
                  in3*(prod_plus_1 - cp3) +
                  in4*(prod_plus_1 - cp4) +
                  in5*(prod_plus_1 - cp5) +
                  in6*(prod_plus_1 - cp6);
  store(scale*result, indx, output);
}

kernel void llor6(input_t data1,
                  input_t data2,
                  input_t data3,
                  input_t data4,
                  input_t data5,
                  input_t data6,
                  float degree_val,
                  char adapt,
                  float scale,
                  output_t output) {
  index_t indx = get_global_index();
  calc_t degree = (calc_t) degree_val;
  
  calc_t in1 = load(indx, data1);
  calc_t in2 = load(indx, data2);
  calc_t in3 = load(indx, data3);
  calc_t in4 = load(indx, data4);
  calc_t in5 = load(indx, data5);
  calc_t in6 = load(indx, data6);
  
  if (adapt) {
    calc_t maxval = fmax(in1, in2);
    maxval = fmax(maxval, in3);
    maxval = fmax(maxval, in4);
    maxval = fmax(maxval, in5);
    maxval = fmax(maxval, in6);
    degree /= maxval;
  }
  
  calc_t cp1 = contpart(-in1, degree);
  calc_t cp2 = contpart(-in2, degree);
  calc_t cp3 = contpart(-in3, degree);
  calc_t cp4 = contpart(-in4, degree);
  calc_t cp5 = contpart(-in5, degree);
  calc_t cp6 = contpart(-in6, degree);
  calc_t prod_plus_1 = cp1*cp2*cp3*cp4*cp5*cp6 + 1.f;
  calc_t result = in1*(prod_plus_1 - cp1) +
                  in2*(prod_plus_1 - cp2) +
                  in3*(prod_plus_1 - cp3) +
                  in4*(prod_plus_1 - cp4) +
                  in5*(prod_plus_1 - cp5) +
                  in6*(prod_plus_1 - cp6);
  store(scale*result, indx, output);
}
