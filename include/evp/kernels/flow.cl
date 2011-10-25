CLIP_STRINGIFY(
kernel void grad2polar(global imval* xs,
                       global imval* ys,
                       global imval* mag,
                       global imval* angle) {
	int indx = get_global_index();
  float x = load(imval, indx, xs);
  float y = load(imval, indx, ys);
  angle[indx] = atan2(x, -y);
  mag[indx] = hypot(x, y);
}

kernel void unitvec(global imval* angles,
                    global imval* vs,
                    global imval* us) {
	int indx = get_global_index();
  float angle = load(imval, indx, angles);
  us[indx] = cos(angle);
  vs[indx] = sin(angle);
}

kernel void ktkn(global imval* us,
                 global imval* vs,
                 global imval* vxs,
                 global imval* vys,
                 global imval* kns,
                 global imval* kts) {
  int indx = get_global_index();
  float u = load(imval, indx, us);
  float v = load(imval, indx, vs);
  float vx = load(imval, indx, vxs);
  float vy = load(imval, indx, vys);
  kts[indx] = (v*vy + u*vx)/u;
  kns[indx] = (u*vy - v*vx)/u;
}

kernel void rescale(global imval* input,
                    float min, float max, float targ_min, float targ_max,
                    int filter, global imval* output) {
  int indx = get_global_index();
  float inval = load(imval, indx, input);
  float outval = targ_min + (inval - min)/(max - min)*(targ_max - targ_min);
  if (filter) outval *= inval > min && inval < max;
  store(imval, outval, indx, output);
}

kernel void flowdiscr(global imval* confs,
                      global imval* thetas,
                      float targ_theta, float theta_step, int npis,
                      global imval* kts,
                      global imval* kns,
                      float targ_kt, float targ_kn, float k_step,
                      global imval* output) {
  int indx = get_global_index();
  
  float conf = load(imval, indx, confs);
  float theta = load(imval, indx, thetas);
  float kt = load(imval, indx, thetas);
  float kn = load(imval, indx, kns);
  
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
  
  store(imval, test ? conf : 0.f, indx, output);
}
)
