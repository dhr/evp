CLIP_STRINGIFY(
kernel void grad2polar(global float *xs,
                       global float *ys,
                       global float *mag,
                       global float *angle) {
	int indx = get_global_index();
  float x = xs[indx];
  float y = ys[indx];
  angle[indx] = atan2(x, -y);
  mag[indx] = hypot(x, y);
}

kernel void unitvec(global float *angles,
                    global float *vs,
                    global float *us) {
	int indx = get_global_index();
  float angle = angles[indx];
  us[indx] = cos(angle);
  vs[indx] = sin(angle);
}

kernel void ktkn(global float *us,
                 global float *vs,
                 global float *vxs,
                 global float *vys,
                 global float *kns,
                 global float *kts) {
  int indx = get_global_index();
  float u = us[indx];
  float v = vs[indx];
  float vx = vxs[indx];
  float vy = vys[indx];
  kts[indx] = (v*vy + u*vx)/u;
  kns[indx] = (u*vy - v*vx)/u;
}

kernel void rescale(global float *input,
                    float min, float max, float targ_min, float targ_max,
                    int filter, global float *output) {
  int indx = get_global_index();
  float inval = input[indx];
  float outval = targ_min + (inval - min)/(max - min)*(targ_max - targ_min);
  if (filter) outval *= inval > min && inval < max;
  output[indx] = outval;
}

#define PI 3.14159265358979323846f

kernel void flowdiscr(global float *confs,
                      global float *thetas,
                      float targ_theta, float theta_step, int npis,
                      global float *kts,
                      global float *kns,
                      float targ_kt, float targ_kn, float k_step,
                      global float *output) {
  int indx = get_global_index();
  
  float conf = confs[indx];
  float theta = thetas[indx];
  float kt = kts[indx];
  float kn = kns[indx];
  
  int test = theta < 0.f;
  theta = test ? theta + npis*PI : theta;
  
  if (npis == 1) {
    kt = test ? -kt : kt;
    kn = test ? -kn : kn;
  }
  
  test = ((fabs(theta - targ_theta) < theta_step/2 ||
           fabs(theta + npis*PI - targ_theta) < theta_step/2 ||
           fabs(theta - npis*PI - targ_theta) < theta_step/2) &&
          fabs(kt - targ_kt) < k_step/2 &&
          fabs(kn - targ_kn) < k_step/2);
  
  output[indx] = test ? conf : 0.f;
}
)
