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

/* Perona-Malik filter */
/*

The Perona-Malik (PM) filter algorithm is a type of edge-preserving, nonlinear diffusion filter used for image
processing. It was developed by Carlo Perona and João Malik in 1990.

The PM filter algorithm is designed to remove noise from an image while preserving its edges. The algorithm works by
diffusing the image over time, with the diffusion process being regulated by a diffusion coefficient that varies
depending on the image gradient. In regions where the gradient is high, such as edges and boundaries, the diffusion
coefficient is small, which limits the amount of diffusion and preserves the edges. In regions where the gradient is
low, such as smooth, homogeneous regions, the diffusion coefficient is large, which allows more diffusion
and helps to reduce noise.

The PM filter algorithm is governed by the following partial differential equation:

∂I/∂t = div(c(x,y,t)∇I)

where I is the image, t is time, div is the divergence operator, ∇ is the gradient operator,
and c(x,y,t) is the diffusion coefficient at position (x,y) and time t. The diffusion coefficient c(x,y,t) is a function
 of the gradient magnitude of the image and a parameter that controls the strength of the diffusion.

The PM filter algorithm is applied iteratively over the image until the desired level of noise reduction is achieved.
The output of the algorithm is a smoothed image that preserves the edges of the original image.

Explanation by chatGPT 2023-02-15

*/

<%include file="constants.mako" />
<%include file="helpers.mako" />

/* Perona-Malik filter for 1,2,3,4 channel images*/
%for planes in ["",2,3,4]:

__kernel 
void filterImage${planes}(int width, int height, FLOAT${planes} *input, float scale, float step_size, FLOAT${planes} *output)
{
    size_t i = get_global_id(0);
    int x = X(i);
    int y = Y(i);
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
      CF = ONE${planes}/(ONE${planes} + (CF * CF));
      FLOAT${planes} accumulator = di * CF;
      di = s-central;
      CF = (di*invscale);
      CF = ONE${planes}/(ONE${planes} + (CF * CF));
      accumulator += di * CF;
      di = e-central;
      CF = (di*invscale);
      CF = ONE${planes}/(ONE${planes} + (CF * CF));
      accumulator += di * CF;
      di = w-central;
      CF = (di*invscale);
      CF = ONE${planes}/(ONE${planes} + (CF * CF));
      accumulator += di * CF;
      accumulator *= step_size;
      central += accumulator;
    }
    output[i] = central;
    barrier(CLK_GLOBAL_MEM_FENCE);
}
%endfor