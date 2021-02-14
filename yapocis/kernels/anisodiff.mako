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
    barrier(CLK_GLOBAL_MEM_FENCE);
}
#define ONES (float2)(1.0f,1.0f)
__kernel
void filterImage2(int width, int height, __global float2 *input, float scale, float step_size, __global float2 *output)
{
    size_t i = get_global_id(0);
    size_t last = width*height;
    int x = i / height;
    int y = i%height;
    int updated = 0;
    float2 scale2 = (float2)(scale,scale);
    // Linear coordinates for y-1,x-1,x+1,y+1
    size_t yb = i-1, xl = i-height, xr = i + height, yt = i+1;
    // and if cell neigborhood is in bounds
    float2 central = input[i];
    float2 invscale = ONES/scale2;
    float2 step_size2 = (float2)(step_size,step_size);
    if ((0<x) && (x<(width-1)) && ((0<y) && (y<(height-1)))) {
      float2 n = input[i+1];
      float2 s = input[i-1];
      float2 e = input[i-height];
      float2 w = input[i+height];
      float2 di = n-central;
      float2 CF = di*invscale;
      CF = ONES/(ONES + (CF * CF));
      float2 accumulator = di * CF;
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
      accumulator *= step_size2;
      central += accumulator;
    }
    output[i] = central;
    barrier(CLK_GLOBAL_MEM_FENCE);
}


#define ONES (float3)(1.0f,1.0f,1.0f)
__kernel
void filterImage3(int width, int height, __global float3 *input, float scale, float step_size, __global float3 *output)
{
    size_t i = get_global_id(0);
    size_t last = width*height;
    int x = i / height;
    int y = i%height;
    int updated = 0;
    float3 scale3 = (float3)(scale,scale,scale);
    // Linear coordinates for y-1,x-1,x+1,y+1
    size_t yb = i-1, xl = i-height, xr = i + height, yt = i+1;
    // and if cell neigborhood is in bounds
    float3 central = input[i];
    float3 invscale = ONES/scale3;
    float3 step_size3 = (float3)(step_size,step_size,step_size);
    if ((0<x) && (x<(width-1)) && ((0<y) && (y<(height-1)))) {
      float3 n = input[i+1];
      float3 s = input[i-1];
      float3 e = input[i-height];
      float3 w = input[i+height];
      float3 di = n-central;
      float3 CF = di*invscale;
      CF = ONES/(ONES + (CF * CF));
      float3 accumulator = di * CF;
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
      accumulator *= step_size3;
      central += accumulator;
    }
    output[i] = central;
    barrier(CLK_GLOBAL_MEM_FENCE);
}


#define ONES (float4)(1.0f,1.0f,1.0f,1.0f)
__kernel
void filterImage4(int width, int height, __global float4 *input, float scale, float step_size, __global float4 *output)
{
    size_t i = get_global_id(0);
    size_t last = width*height;
    int x = i / height;
    int y = i%height;
    int updated = 0;
    float4 scale4 = (float4)(scale,scale,scale,scale);
    // Linear coordinates for y-1,x-1,x+1,y+1
    size_t yb = i-1, xl = i-height, xr = i + height, yt = i+1;
    // and if cell neigborhood is in bounds
    float4 central = input[i];
    float4 invscale = ONES/scale4;
    float4 step_size4 = (float4)(step_size,step_size,step_size,step_size);
    if ((0<x) && (x<(width-1)) && ((0<y) && (y<(height-1)))) {
      float4 n = input[i+1];
      float4 s = input[i-1];
      float4 e = input[i-height];
      float4 w = input[i+height];
      float4 di = n-central;
      float4 CF = di*invscale;
      CF = ONES/(ONES + (CF * CF));
      float4 accumulator = di * CF;
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
      accumulator *= step_size4;
      central += accumulator;
    }
    output[i] = central;
    barrier(CLK_GLOBAL_MEM_FENCE);
}
