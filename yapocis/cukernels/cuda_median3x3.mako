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

/* Kernel to compute 3x3 median filter on a 2D image */

static inline float median9(float *a, int n) {
    float l,h;
    bool lt;
    % for i in range(9//2+1):
        % for j in range(i+1, 9):
            lt = a[${j}] < a[${i}];
            l = lt ? a[${j}] : a[${i}];
            h = lt ? a[${i}] : a[${j}];
            a[${i}] = l;
            a[${j}] = h;
        %endfor
    %endfor
    return a[n/2];
}


__kernel
void median3x3(int width, int height, float* a, float* ret )
{
    float aa[9];
    size_t i;
    float m;
    size_t left = height + 1;
    size_t right = width - left;
    int h = height;
    i = get_global_id(0);
    m = a[i];
    if ((i >= left) &&  (i < right)) {
		aa[0] =  a[(i-h)-1];
		aa[1] =  a[i-h];
		aa[2] =  a[(i-h)+1];
		aa[3] =  a[i-1];
		aa[4] =  a[i];
		aa[5] =  a[i+1];
		aa[6] =  a[(i+h)-1];
		aa[7] =  a[i+h];
		aa[8] =  a[(i+h)+1];
	    m = median9(&aa[0], 9);
    }
    ret[i] = m;
    return;
}