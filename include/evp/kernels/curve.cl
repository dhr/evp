CLIP_STRINGIFY(
kernel void stabilize(global float *input1,
                      global float *slice_sum,
                      int n, float stab_amt,
                      global float *output) {
  int indx = get_global_index();
  float inval = input1[indx];
  output[indx] = inval + stab_amt*(fabs(inval) - slice_sum[indx]/n);
}

kernel void surround2(global float *data1,
                      global float *data2,
                      float degree,
                      global float *output) {
  int indx = get_global_index();
  
  float maxval = fmax(fabs(data1[indx]), fabs(data2[indx]));
  if (maxval == 0) maxval = 1;
  
  float part = smoothpart(data1[indx], degree/maxval);
  float pos = part;
  float neg = 1 - part;
  float ambig = 2*pos*neg;
  
  part = smoothpart(data2[indx], degree/maxval);
  pos *= part;
  neg *= (1 - part);
  
  output[indx] = data1[indx] + data2[indx]*(pos + neg + ambig);
}
)
