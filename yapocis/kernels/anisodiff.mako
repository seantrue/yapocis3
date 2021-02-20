/* Perona-Malik filter */

#define ZERO 0.0f
#define ZERO2 (float2)(ZERO,ZERO)
#define ZERO3 (float3)(ZERO,ZERO,ZERO)
#define ZERO4 (float4)(ZERO,ZERO,ZERO,ZERO)
#define ONE 1.0f
#define ONE2 (float2)(ONE,ONE)
#define ONE3 (float3)(ONE,ONE,ONE)
#define ONE4 (float4)(ONE,ONE,ONE,ONE)
#define FLOAT float
#define FLOAT2 float2
#define FLOAT3 float3
#define FLOAT4 float4

%for planes in ["",2,3,4]:
#define ZEROS ZERO${planes}
#define ONES ONE${planes}
#define FLOAT float${planes}

__kernel
void filterImage${planes}(int width, int height, __global FLOAT${planes} *input, float scale, float step_size, __global FLOAT${planes} *output)
{
    size_t i = get_global_id(0);
    size_t last = width*height;
    int x = i / height;
    int y = i%height;
    FLOAT${planes} CF;
    FLOAT${planes} central = input[i];
    FLOAT${planes} invscale = 1.0f/scale;
    if ((0<x) && (x<(width-1)) && ((0<y) && (y<(height-1)))) {
      FLOAT${planes} n = input[i+1];
      FLOAT${planes} s = input[i-1];
      FLOAT${planes} e = input[i-height];
      FLOAT${planes} w = input[i+height];
      FLOAT${planes} di = n-central;
      CF = di*invscale;
      CF = ONES/(ONES + (CF * CF));
      FLOAT${planes} accumulator = di * CF;
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
