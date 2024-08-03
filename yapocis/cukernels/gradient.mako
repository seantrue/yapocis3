/* Kernels to compute direction and magnitude of gradient of an image. */

// Include necessary CUDA headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// Assuming constants.cuh and helpers.cuh exist with equivalent CUDA definitions
#include "constants.cuh"
#include "helpers.cuh"

// Helper function to compute atan2 divided by PI
__device__ float atan2pi(float y, float x) {
    return atan2f(y, x) / M_PI;
}

// Template for different number of channels
template<int planes>
__global__ void gradient(int width, int height, float* input, int reach, float* grad, float* theta)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int x = X(i);
    int y = Y(i);

    // Use float1, float2, float3, or float4 based on planes
    typedef typename std::conditional<planes == 1, float1,
             typename std::conditional<planes == 2, float2,
             typename std::conditional<planes == 3, float3,
                                                    float4>::type>::type>::type FloatType;

    FloatType gr = make_float<planes>(0.0f);
    FloatType th = make_float<planes>(0.0f);

    if (0 <= (x-reach) && (x+reach) < width && 0 <= (y-reach) && (y+reach) < height) {
        FloatType dx, dy;
        for (int p = 0; p < planes; ++p) {
            dx.f[p] = input[ADDRESS(x+reach,y)*planes + p] - input[ADDRESS(x-reach,y)*planes + p];
            dy.f[p] = input[ADDRESS(x,y+reach)*planes + p] - input[ADDRESS(x,y-reach)*planes + p];
        }

        for (int p = 0; p < planes; ++p) {
            gr.f[p] = sqrtf(dx.f[p]*dx.f[p] + dy.f[p]*dy.f[p]);
            th.f[p] = atan2pi(dy.f[p], dx.f[p]);
        }
    }

    for (int p = 0; p < planes; ++p) {
        grad[i*planes + p] = isnan(gr.f[p]) ? 0.0f : gr.f[p];
        theta[i*planes + p] = isnan(th.f[p]) ? 0.0f : th.f[p];
    }
}

// Explicit instantiations for 1, 2, 3, and 4 planes
template __global__ void gradient<1>(int, int, float*, int, float*, float*);
template __global__ void gradient<2>(int, int, float*, int, float*, float*);
template __global__ void gradient<3>(int, int, float*, int, float*, float*);
template __global__ void gradient<4>(int, int, float*, int, float*, float*);
