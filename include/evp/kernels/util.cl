calc_t contpart(calc_t x, calc_t degree);
calc_t smoothpart(calc_t x, calc_t degree);

calc_t contpart(calc_t x, calc_t degree) {
	calc_t invDeg = 1/degree;
  x += invDeg/2;
  
  return iif(x <= (calc_t) 0, (calc_t) 0,
             iif(x >= invDeg, (calc_t) 1, degree*x));
}

calc_t smoothpart(calc_t x, calc_t degree) {
	x *= degree;
  
  calc_t temp = exp(-1/(0.5f + x));
  temp = temp/(temp + exp(-1/(0.5f - x)));
  
  return iif(x <= (calc_t) -0.5, (calc_t) 0,
             iif(x >= (calc_t) 0.5, (calc_t) 1, temp));
}
