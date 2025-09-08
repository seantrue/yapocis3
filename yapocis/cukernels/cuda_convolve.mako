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

/* One dimensional convolution with a kernel conv that is determined at compile time */

<% left = len(conv)//2 %>
#define FAST 1000
__kernel
void ${name}(int width,  float* a, float* ret )
{
    size_t i = get_global_id(0);
    float sum;
    size_t right = width-${left};
    sum = 0.0;
    if ((i > (size_t)${left}) &&  (i < right)) {
	%for j in range(len(conv)):
       sum = sum + a[i+(${j-left})] * ${conv[j]};
    %endfor
    }
    ret[i] = sum;
    return;
}

__kernel void ${name}_fast(int width,  float* a, float* ret, int chunk)
{
    size_t i = get_global_id(0) * chunk;
    float sum;
    size_t right = width-${left};
    volatile __local float buffer[FAST];
    for (int j = 0; j < ${len(conv)}; j++) {
        buffer[j] = i+j >= width ? 0.0 : a[i+j];
    }
    for (int j = 0; j < chunk-${len(conv)}; j++) {
        sum = 0.0;
        int ipixel = i+j >= right ? right-1 : i+j;
        if ((i > (size_t)${left}) &&  (i < right)) {
        %for j in range(len(conv)):
           sum = sum + a[j] * ${conv[j]};
        %endfor
        }
        ret[ipixel] = sum;
    }
    return;
}
