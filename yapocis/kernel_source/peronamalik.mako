/* Perona-Malik filter */

__kernel 
void filterImage(int width, int height, __global float *input, float scale, float step_size, __global float *output) 
{
    size_t i = get_global_id(0);
    size_t last = width*height;
    int x = i / height;
    int y = i%height;
    int updated = 0;
    float CF;
    // Linear coordinates for y-1,x-1,x+1,y+1
    size_t yb = i-1, xl = i-height, xr = i + height, yt = i+1;
    // and if cell neigborhood is in bounds
    float central = input[i];
    if ((0<x) && (x<(width-1)) && ((0<y) && (y<(height-1)))) {
      float n = input[i+1];
      float s = input[i-1];
      float e = input[i-height];
      float w = input[i+height];
      float di = n-central;
      float invscale = 1.0f/scale;
      CF = di*invscale;
      CF = 1.0/(1.0 + (CF * CF));
      float accumulator = di * CF;
      di = s-central;
      CF = (di*invscale);
      CF = 1.0/(1.0 + (CF * CF));
      accumulator += di * CF;
      di = e-central;
      CF = (di*invscale);
      CF = 1.0/(1.0 + (CF * CF));
      accumulator += di * CF;
      di = w-central;
      CF = (di*invscale);
      CF = 1.0/(1.0 + (CF * CF));
      accumulator += di * CF;
      accumulator *= step_size;
      central += accumulator;
    }
    output[i] = central;
}
