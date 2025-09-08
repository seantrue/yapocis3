/* Auto-translated from OpenCL to CUDA (Mako kept intact). */
#ifndef YAPOCIS_CUDA_SHIMS_H
#define YAPOCIS_CUDA_SHIMS_H
#define __kernel extern "C" __global__
#define __local  __shared__
#define barrier(x) __syncthreads()
__device__ __forceinline__ int get_global_id_x(){return blockIdx.x*blockDim.x+threadIdx.x;}
__device__ __forceinline__ int get_global_id_y(){return blockIdx.y*blockDim.y+threadIdx.y;}
__device__ __forceinline__ int get_global_id_z(){return blockIdx.z*blockDim.z+threadIdx.z;}
#define get_global_id(N) ( (N)==0 ? get_global_id_x() : ( (N)==1 ? get_global_id_y() : get_global_id_z() ) )
__device__ __forceinline__ int clampi(int v,int lo,int hi){return max(lo,min(hi,v));}
__device__ __forceinline__ float clampf(float v,float lo,float hi){return fminf(hi,fmaxf(lo,v));}
#define clamp(V,L,H) _Generic((V), float:clampf, double:clampf, default:clampi)((V),(L),(H))
template<typename V> __device__ __forceinline__ V  vload(int off, const void* base){return reinterpret_cast<const V*>(base)[off];}
template<typename V> __device__ __forceinline__ void vstore(V v, int off, void* base){reinterpret_cast<V*>(base)[off]=v;}
#define vload4(OFF,PTR)       vload<float4>((OFF),(const void*)(PTR))
#define vstore4(VAL,OFF,PTR)  vstore<float4>((VAL),(OFF),(void*)(PTR))
#endif

// Kernels for operating on HxWx4 arrays
// array[,,3] is the "alpha" channel
// The alpha channel is ignored (but must be present) for the input/output array (a)
// Colors are [N,4] where color[,3] is the alpha and is _not_ ignored

<%include file="constants.mako" />
#define DXDY2
<%include file="helpers.mako" />

#define ONES ONE4
#define ZEROS ZERO4

__kernel void lines(int width, int height, READWRITE float4 *a, int nelements,
                    READONLY float2 *coords, READONLY float *rdist,
                    READONLY float4 *colors) {
  size_t i = get_global_id(0);
  int x = X(i);
  int y = Y(i);
  float4 color = a[i];

  float2 point = (float2)((float)x, (float)y);
  for (int ielement = 0; ielement < nelements; ielement++) {
    float threshold2 = ((*rdist++ * .5) + .5);
    threshold2 *= threshold2;
    float2 linestart = *coords++;
    float2 lineend = *coords++;
    float4 rgba = *colors++;
    float4 alpha = (float4)(rgba.s3, rgba.s3, rgba.s3, 0.0f);
    float linemag2 = dxdy2(lineend, linestart);
    float2 delta = lineend - linestart;
    float2 u2 = (point - linestart) * (delta);
    float u = (u2.x + u2.y) / linemag2;
    float2 intersection = linestart + (u * delta);
    float dist = dxdy2(point,intersection);
    float4 clr = (rgba * alpha) + (color * (ONES - alpha));
    color = (u >= 0.0f && u <= 1.0f && dist < threshold2) ? clr : color;
  }
  color.s3 = 0.0f;
  a[i] = color;
  barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void rects(int width, int height, READWRITE float4 *a, int nelements,
                    READONLY float4 *coords, READONLY float4 *colors) {
  size_t i = get_global_id(0);
  float4 pixel = a[i];
  int x = X(i);
  int y = Y(i);

  for (int ielement = 0; ielement < nelements; ielement++) {
    float4 rgba = *colors++;
    float4 alpha = (float4)(rgba.s3, rgba.s3, rgba.s3, 0.0f);
    float4 clr = (rgba * alpha) + (pixel * (ONES - alpha));

    float4 region = *coords++;

    float l = min(region.x, region.z);
    float b = min(region.y, region.w);
    float r = max(region.x, region.z);
    float t = max(region.y, region.w);
    pixel = (l <= x && x <= r && b <= y && y <= t) ? clr : pixel;
  }
  a[i] = pixel;
  barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void circles(int width, int height, READWRITE float4 *a, int nelements,
                      READONLY float *coords, READONLY float4 *colors) {
  size_t i = get_global_id(0);
  float4 pixel = a[i];
    int x = X(i);
    int y = Y(i);

    float2 point = (float2)((float)x, (float)y);
      for (int ielement = 0; ielement < nelements; ielement++) {
        float4 rgba = *colors++;
        float4 alpha = (float4)(rgba.s3, rgba.s3, rgba.s3, 0.0f);
        float4 clr = (rgba * alpha) + (pixel * (ONES - alpha));

        float x = *coords++;
        float y = *coords++;
        float radius = *coords++;
        radius *= radius;

        float2 center = (float2)(x, y);
        pixel = dxdy2(center, point) <= radius ? clr : pixel;
      }
  a[i] = pixel;
  barrier(CLK_GLOBAL_MEM_FENCE);
}

#define signex(p1, p2, p3)                                                     \
  ((p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y))

__kernel void triangles(int width, int height, READWRITE float4 *a, int nelements,
                        READONLY float2 *coords, READONLY float4 *colors) {
  size_t i = get_global_id(0);
  float4 pixel = a[i];
  int x = X(i);
  int y = Y(i);
  float2 point = (float2)((float)x, (float)y);
  for (int ielement = 0; ielement < nelements; ielement++) {
    float4 rgba = *colors++;
    float4 alpha = (float4)(rgba.s3, rgba.s3, rgba.s3, 0.0f);
    float4 clr = (rgba * alpha) + (pixel * (ONES - alpha));

    float2 v1 = *coords++;
    float2 v2 = *coords++;
    float2 v3 = *coords++;

    float d1 = signex(point, v1, v2);
    float d2 = signex(point, v2, v3);
    float d3 = signex(point, v3, v1);
    bool has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    bool has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    pixel = !(has_neg && has_pos) ? clr : pixel;
  }
  a[i] = pixel;
  barrier(CLK_GLOBAL_MEM_FENCE);
}

// (x-h)^2/a^2 + (y-k)^2/b^2 <= 1
__kernel void ellipses(int width, int height, READWRITE float4 *a, int nelements,
                       READONLY float4 *coords, READONLY float4 *colors) {
  size_t i = get_global_id(0);
  float4 pixel = a[i];
  int x = X(i);
  int y = Y(i);
  float2 point = (float2)((float)x, (float)y);
  for (int ielement = 0; ielement < nelements; ielement++) {
    float4 rgba = *colors++;
    float4 alpha = (float4)(rgba.s3, rgba.s3, rgba.s3, 0.0f);
    float4 clr = (rgba * alpha) + (pixel * (ONES - alpha));

    float4 center_axes = *coords++;
    float2 center = (float2)(center_axes.s0,center_axes.s1);
    float2 axes = (float2)(center_axes.s2,center_axes.s3);
    float2 dist = (point - center) * (point - center) / (axes * axes);

    pixel = (dist.x + dist.y) <= 1.0 ? clr : pixel;
  }
  a[i] = pixel;
  barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void composite(int width, int height, READWRITE float4 *a, int nelements,
                        READONLY int4 *coords, READWRITE float4 *pixels,
                        READONLY int *ipixels, int npixels) {
  size_t i = get_global_id(0);
  float4 pixel = a[i];
  int x = X(i);
  int y = Y(i);

  int2 xy = (int2)(x,y);
  for (int ielement = 0; ielement < nelements; ielement++) {
    // Get the x,y,w,h of the destination
    int4 coord = *coords++;
    int ipixel = *ipixels++;

    int2 lb = coord.s10;
    int2 yt = lb + coord.s32;
    // Is the current xy inside the destination area?
    int outside = ANYISLESS(xy,lb) + ANYISGTE(xy,yt);
    int2 source_location = xy - lb;
    int source_offset = (source_location.x * coord.s2) + source_location.y;
    int ipixel_source = ipixel + source_offset;
    int inside = !outside && 0 <= ipixel_source && ipixel_source < npixels;

    float4 rgba = inside ? pixels[ipixel_source] : ZEROS;
    float4 alpha = (float4)(rgba.s3, rgba.s3, rgba.s3, 0.0f);
    float4 clr = (rgba * alpha) + (pixel * (ONES - alpha));
    pixel = inside ? clr : pixel;
  }
  a[i] = pixel;
  barrier(CLK_GLOBAL_MEM_FENCE);
}
