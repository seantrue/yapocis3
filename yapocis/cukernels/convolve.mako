<%
import numpy as np
# Assuming conv is defined here, e.g.:
# conv = [0.1, 0.2, 0.4, 0.2, 0.1]
left = len(conv) // 2
%>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define FAST 1000

__constant__ float d_conv[${len(conv)}];  // Device constant memory for conv

__global__ void ${name}(int width, float* a, float* ret)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    int right = width - ${left};

    if ((i > ${left}) && (i < right)) {
        % for j in range(len(conv)):
        sum += a[i + (${j-left})] * d_conv[${j}];
        % endfor
    }
    ret[i] = sum;
}

__global__ void ${name}_fast(int width, float* a, float* ret, int chunk)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * chunk;
    float sum;
    int right = width - ${left};
    __shared__ float buffer[FAST];

    for (int j = 0; j < ${len(conv)}; j++) {
        buffer[j] = i + j >= width ? 0.0f : a[i + j];
    }
    __syncthreads();

    for (int j = 0; j < chunk - ${len(conv)}; j++) {
        sum = 0.0f;
        int ipixel = min(i + j, right - 1);
        if ((i > ${left}) && (i < right)) {
            % for k in range(len(conv)):
            sum += buffer[j + ${k}] * d_conv[${k}];
            % endfor
        }
        ret[ipixel] = sum;
    }
}

// Host function to set up and launch kernels
void launch_kernels(int width, float* h_a, float* h_ret, int chunk)
{
    float *d_a, *d_ret;
    int size = width * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_ret, size);

    // Copy input data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    // Copy conv to constant memory
    float conv[] = {${", ".join(map(str, conv))}};
    cudaMemcpyToSymbol(d_conv, conv, ${len(conv)} * sizeof(float));

    // Launch kernels
    int block_size = 256;
    int grid_size = (width + block_size - 1) / block_size;

    ${name}<<<grid_size, block_size>>>(width, d_a, d_ret);
    ${name}_fast<<<grid_size / chunk, block_size>>>(width, d_a, d_ret, chunk);

    // Copy results back to host
    cudaMemcpy(h_ret, d_ret, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_ret);
}