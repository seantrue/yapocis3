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

/* Kernels to compute direction and magnitude of gradient of an image. */

<%include file="constants.mako" />
<%include file="helpers.mako" />

/* Compute gradient for images with 1,2,3,4 channels per pixel. */

%for planes in ["",2,3,4]:
__kernel
void gradient${planes}(int width, int height, FLOAT${planes}* input, int reach, FLOAT${planes}* grad, FLOAT${planes} *theta )
{
    size_t i = get_global_id(0);
    int x = X(i);
    int y = Y(i);
    FLOAT${planes} gr = ZERO${planes};
    FLOAT${planes} th = ZERO${planes};
    if (0 <= (x-reach)  && (x+reach) < width && 0 <= (y-reach) && (y+reach) < height) {
        FLOAT${planes} dx = input[ADDRESS(x+reach,y)] - input[ADDRESS(x-reach,y)];
        FLOAT${planes} dy = input[ADDRESS(x,y+reach)] - input[ADDRESS(x,y-reach)];
        gr = sqrt(dx*dx+dy*dy);
        th = atan2pi(dy, dx);
    }
    grad[i] = isnan(gr) ? ZERO${planes}: gr;
    theta[i] = isnan(th) ? ZERO${planes}: th;
}
%endfor