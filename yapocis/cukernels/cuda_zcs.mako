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

/* Kernels to find zero-crossings in an image. */

<%include file="constants.mako" />
<%include file="helpers.mako" />

/* Find zero-crossings for pixels of 1-4 channels. */
%for planes in [1,2,3,4]:
#define ADDRESS${planes}(xx,yy) (ADDRESS(xx,yy))*${planes}

__kernel
void zcs${planes}(int width, int height, float threshold, float* input, float* zcs )
{
    size_t i = get_global_id(0)*${planes};
    int x = max((size_t)1,X(i));
    x = min(x,width-2);
    int y = max((size_t)1,Y(i));
    y = min(y, height-2);
    float zclt;
    float v, below, left, above, right;
    float delta;
    %for plane in range(planes):
    v = input[i];
    above = input[ADDRESS${planes}(x,y+1)];
    below = input[ADDRESS${planes}(x,y-1)];
    left = input[ADDRESS${planes}(x-1,y)];
    right = input[ADDRESS${planes}(x+1,y)];
    delta = ((above-below)*(above-below)+(right-left)*(right-left));
    above = sign(above);
    below = sign(below);
    left = sign(left);
    right = sign(right);
    zclt = (delta > threshold) && ((below != above) || (left != right)) ? ONE : ZERO;
    zcs[i] = zclt;
    i += 1;
    %endfor
}
%endfor

__kernel
void sub(float* input1, float* input2, float *output)
{
    size_t i;
    i = get_global_id(0);
    output[i] = input1[i] - input2[i];
}
