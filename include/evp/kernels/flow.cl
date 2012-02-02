kernel void grad2polar(input_t xs,
                       input_t ys,
                       output_t mag,
                       output_t angle) {
	index_t indx = get_global_index();
  calc_t x = load(indx, xs);
  calc_t y = load(indx, ys);
  store(atan2(x, -y), indx, angle);
  store(hypot(x, y), indx, mag);
}

kernel void unitvec(input_t angles,
                    output_t vs,
                    output_t us) {
	index_t indx = get_global_index();
  calc_t angle = load(indx, angles);
  store(cos(angle), indx, us);
  store(sin(angle), indx, vs);
}

kernel void ktkn(input_t us,
                 input_t vs,
                 input_t vxs,
                 input_t vys,
                 output_t kns,
                 output_t kts) {
  index_t indx = get_global_index();
  calc_t u = load(indx, us);
  calc_t v = load(indx, vs);
  calc_t vx = load(indx, vxs);
  calc_t vy = load(indx, vys);
  store((v*vy + u*vx)/u, indx, kts);
  store((u*vy - v*vx)/u, indx, kns);
}

kernel void rescale(input_t input,
                    float min, float max,
                    float targ_min, float targ_max,
                    int filter,
                    output_t output) {
  index_t indx = get_global_index();
  calc_t inval = load(indx, input);
  calc_t outval = targ_min + (inval - min)/(max - min)*(targ_max - targ_min);
  if (filter) {
    outval = iif(inval > (calc_t) min,
                 iif(inval < (calc_t) max, outval, (calc_t) 0),
                 (calc_t) 0);
  }
  store(outval, indx, output);
}

kernel void flowdiscr(input_t confs,
                      input_t thetas,
                      float targ_theta, float theta_step, int npis,
                      input_t kts,
                      input_t kns,
                      float targ_kt, float targ_kn, float k_step,
                      output_t output) {
  index_t indx = get_global_index();
  
  calc_t conf = load(indx, confs);
  calc_t theta = load(indx, thetas);
  calc_t kt = load(indx, kts);
  calc_t kn = load(indx, kns);
  
  bool_t test = theta < (calc_t) 0.f;
  theta = iif(test, theta + npis*PI, theta);
  
  if (npis == 1) {
    kt = iif(test, -kt, kt);
    kn = iif(test, -kn, kn);
  }
  
  test = ((fabs(theta - targ_theta) < theta_step/2 ||
           fabs(theta + npis*PI - targ_theta) < theta_step/2 ||
           fabs(theta - npis*PI - targ_theta) < theta_step/2) &&
          fabs(kt - targ_kt) < k_step/2 &&
          fabs(kn - targ_kn) < k_step/2);
  
  store(iif(test, conf, (calc_t) 0.f), indx, output);
}
