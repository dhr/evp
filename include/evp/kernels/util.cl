calc_t contpart(calc_t x, calc_t degree);
calc_t smoothpart(calc_t x, calc_t degree);

calc_t contpart(calc_t x, calc_t degree) {
	calc_t invDeg = 1.f/degree;
  x += invDeg/2.f;
  
  return iif(x <= 0.f, 0.f, iif(x >= invDeg, 1.f, degree*x));
}

calc_t smoothpart(calc_t x, calc_t degree) {
	x *= degree;
  
  calc_t temp = exp(-1.f/(0.5f + x));
  temp = temp/(temp + exp(-1.f/(0.5f - x)));
  
  return iif(x <= -0.5f, 0.f, iif(x >= 0.5f, 1.f, temp));
}
