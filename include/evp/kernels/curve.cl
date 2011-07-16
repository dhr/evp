CLIP_STRINGIFY(
kernel void stabilize(global imval* input1,
                      global imval* slice_sum,
                      int n, float stab_amt,
                      global imval* output) {
  int indx = get_global_index();
  float inval = load(imval, indx, input1);
  float slice_sum_val = load(imval, indx, slice_sum);
  store(imval, inval + stab_amt*(fabs(inval) - slice_sum_val/n), indx, output);
}

kernel void surround2(global imval* data1,
                      global imval* data2,
                      float degree,
                      global imval* output) {
  int indx = get_global_index();
  
  float in1 = load(imval, indx, data1);
  float in2 = load(imval, indx, data2);
  
  float maxval = fmax(fabs(in1), fabs(in2));
  if (maxval == 0) maxval = 1;
  
  float part = smoothpart(in1, degree/maxval);
  float pos = part;
  float neg = 1 - part;
  float ambig = 2*pos*neg;
  
  part = smoothpart(in2, degree/maxval);
  pos *= part;
  neg *= (1 - part);
  
  store(imval, in1 + in2*(pos + neg + ambig), indx, output);
}
)
