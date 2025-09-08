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

// kernels imageops
__kernel void clamp_pixels(float *input, float low, float high, float *output) {
    size_t i = get_global_id(0);
    float value = input[i];
    value = value < low ? low:  value;
    value = value > high ? high : value;
    output[i] = value;

}
__kernel void stretch(float *input, float input_min, float input_max, float low, float high, float *output) {
    size_t i = get_global_id(0);
    float value = input[i];
    value = value < input_min ? input_min : value;
    value = value > input_max ? input_max : value;
    value = (value - input_min)/(input_max-input_min);
    value = low + value * (high-low);
    output[i] = value;
}

__kernel void binarize(float *input, float threshold, float *output) {
    size_t i = get_global_id(0);
    float value = input[i];
    value = value >= threshold ? 1.0 : 0.0;
    output[i] = value;
}

__kernel void histogram(float *input, int size_of_input, float low, float high, int *hst, int size_of_hst) {
    int n_bins = size_of_hst;
    size_t i = get_global_id(0);
    float value = input[i];
    value = value < low ? low : value;
    value = value > high ? high : value;
    size_t bucket = (size_t)((n_bins*value)/(high-low));
    atomic_inc(hst+bucket);
    barrier(CLK_GLOBAL_MEM_FENCE);

}



