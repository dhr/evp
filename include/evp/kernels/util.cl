CLIP_STRINGIFY(
float contpart(float x, float degree);
float smoothpart(float x, float degree);

float contpart(float x, float degree) {
	float invDeg = 1/degree;
  x += invDeg/2;
  
  if (x <= 0) return 0;
  if (x >= invDeg) return 1;
  return degree*x;
}

float smoothpart(float x, float degree) {
	x *= degree;
  
  if (x <= -0.5) return 0;
  if (x >= 0.5) return 1;
  
  float temp = exp(-1/(0.5f + x));
  return temp/(temp + exp(-1/(0.5f - x)));
}
)
