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

/* Separable convolution kernels for OpenCL, with kernel names and constants determined at compile time */
% for name, conv in convs:
<% margin = len(conv)//2 %>
__kernel
void ${name}(int vertical, int width, int height, float* a, float* ret )
{
    size_t i = get_global_id(0);
    size_t n = width*height;
    float sum = 0.0;
    if (!vertical && ((size_t)(${margin}*height) < i) && (i < (n-${margin}*height))){
       // Horizontal convolution
  	%for j in range(len(conv)):
		sum = sum + a[i+(${j-margin}*height)] * ${conv[j]};
	%endfor
    }
    if (vertical && (${margin} < i) && (i < (n-${margin}))) {
       // Vertical convolution
  	%for j in range(len(conv)):
		sum = sum + a[i+(${j-margin})] * ${conv[j]};
	%endfor
    }
    ret[i] = sum;
    return;
}
__kernel
void ${name}_h(int width, int height, float* a, float* ret )
{
    size_t i = get_global_id(0);
    size_t n = width*height;
    float sum = 0.0;

    // Horizontal convolution
    int inside = (((size_t)(${margin}*height) < i) && (i < (n-${margin}*height)));
  	%for j in range(len(conv)):
    sum += inside ? a[i+(${j-margin}*height)] * ${conv[j]} : 0.0f;
	%endfor
    ret[i] = sum;
    return;
}
__kernel
void ${name}_v(int width, int height, float* a, float* ret )
{
    size_t i = get_global_id(0);
    size_t n = width*height;

    // Vertical convolution
    float sum = 0.0;
    int inside = (${margin} < i) && (i < (n-${margin}));
  	%for j in range(len(conv)):
    sum +=  inside ?  a[i+(${j-margin})] * ${conv[j]} : 0.0;
	%endfor

    ret[i] = sum;
    return;
}
% endfor