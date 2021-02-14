/* Perona-Malik filter */

#define ZERO 0.0f
#define ZERO2 (float2)(ZERO,ZERO)
#define ZERO3 (float3)(ZERO,ZERO,ZERO)
#define ZERO4 (float4)(ZERO,ZERO,ZERO,ZERO)
#define ONE 1.0f
#define ONE2 (float2)(ONE,ONE)
#define ONE3 (float3)(ONE,ONE,ONE)
#define ONE4 (float4)(ONE,ONE,ONE,ONE)

%for planes in ["",2,3,4]:
#define ZEROS ZERO${planes}
#define ONES ONE${planes}
#define FLOAT float${planes}

__kernel 
void filterImage${planes}(int width, int height, __global FLOAT *input, float scale, float step_size, __global FLOAT *output)
{
    size_t i = get_global_id(0);
    size_t last = width*height;
    int x = i / height;
    int y = i%height;
    int updated = 0;
    FLOAT CF;
    // Linear coordinates for y-1,x-1,x+1,y+1
    size_t yb = i-1, xl = i-height, xr = i + height, yt = i+1;
    // and if cell neigborhood is in bounds
    FLOAT central = input[i];
    FLOAT invscale = 1.0f/scale;
    if ((0<x) && (x<(width-1)) && ((0<y) && (y<(height-1)))) {
      FLOAT n = input[i+1];
      FLOAT s = input[i-1];
      FLOAT e = input[i-height];
      FLOAT w = input[i+height];
      FLOAT di = n-central;
      CF = di*invscale;
      CF = ONES/(ONES + (CF * CF));
      FLOAT accumulator = di * CF;
      di = s-central;
      CF = (di*invscale);
      CF = ONES/(ONES + (CF * CF));
      accumulator += di * CF;
      di = e-central;
      CF = (di*invscale);
      CF = ONES/(ONES + (CF * CF));
      accumulator += di * CF;
      di = w-central;
      CF = (di*invscale);
      CF = ONES/(ONES + (CF * CF));
      accumulator += di * CF;
      accumulator *= step_size;
      central += accumulator;
    }
    output[i] = central;
    barrier(CLK_GLOBAL_MEM_FENCE);
}
%endfor