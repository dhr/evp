kernel void stabilize(input_t input1,
                      input_t slice_sum,
                      int n, float stab_amt,
                      output_t output) {
  index_t indx = get_global_index();
  calc_t inval = load(indx, input1);
  calc_t slice_sum_val = load(indx, slice_sum);
  store(inval + stab_amt*(fabs(inval) - slice_sum_val/n), indx, output);
}

kernel void surround2(input_t data1,
                      input_t data2,
                      float degree,
                      output_t output) {
  index_t indx = get_global_index();
  
  calc_t in1 = load(indx, data1);
  calc_t in2 = load(indx, data2);
  
  calc_t maxval = fmax(fabs(in1), fabs(in2));
  maxval = iif(maxval == 0, 1, maxval);
  
  calc_t part = smoothpart(in1, degree/maxval);
  calc_t pos = part;
  calc_t neg = 1 - part;
  calc_t ambig = 2*pos*neg;
  
  part = smoothpart(in2, degree/maxval);
  pos *= part;
  neg *= (1 - part);
  
  store(in1 + in2*(pos + neg + ambig), indx, output);
}
