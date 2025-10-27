// VGG16 Metal Compute Shaders

<%include file="constants.mako" />
<%include file="helpers.mako" />

#include <metal_stdlib>
using namespace metal;

// 1. Convolution Kernel (3x3 with padding=1, stride=1)
kernel void conv2d_3x3(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& input_height [[buffer(4)]],
    constant uint& input_width [[buffer(5)]],
    constant uint& input_channels [[buffer(6)]],
    constant uint& output_channels [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint out_h = gid.x;
    uint out_w = gid.y;
    uint out_c = gid.z;
    
    if (out_h >= input_height || out_w >= input_width || out_c >= output_channels)
        return;
    
    float sum = bias[out_c];
    
    // 3x3 convolution with padding
    for (uint in_c = 0; in_c < input_channels; in_c++) {
        for (uint kh = 0; kh < 3; kh++) {
            for (uint kw = 0; kw < 3; kw++) {
                int in_h = int(out_h + kh) - 1;  // padding = 1
                int in_w = int(out_w + kw) - 1;  // padding = 1
                
                // Check bounds (zero padding)
                if (in_h >= 0 && in_h < int(input_height) && 
                    in_w >= 0 && in_w < int(input_width)) {
                    
                    uint input_idx = in_c * input_height * input_width + 
                                    uint(in_h) * input_width + uint(in_w);
                    uint weight_idx = out_c * input_channels * 9 + 
                                     in_c * 9 + kh * 3 + kw;
                    
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }
    
    uint output_idx = out_c * input_height * input_width + 
                      out_h * input_width + out_w;
    output[output_idx] = sum;
}

// 2. ReLU Activation Kernel
kernel void relu_activation(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size)
        return;
    
    output[gid] = max(0.0f, input[gid]);
}

// 3. Max Pooling Kernel (2x2 with stride=2)
kernel void max_pool_2x2(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& input_height [[buffer(2)]],
    constant uint& input_width [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint out_h = gid.x;
    uint out_w = gid.y;
    uint c = gid.z;
    
    uint output_height = input_height / 2;
    uint output_width = input_width / 2;
    
    if (out_h >= output_height || out_w >= output_width || c >= channels)
        return;
    
    float max_val = -INFINITY;
    
    // 2x2 pooling window
    for (uint kh = 0; kh < 2; kh++) {
        for (uint kw = 0; kw < 2; kw++) {
            uint in_h = out_h * 2 + kh;
            uint in_w = out_w * 2 + kw;
            
            uint input_idx = c * input_height * input_width + 
                            in_h * input_width + in_w;
            
            max_val = max(max_val, input[input_idx]);
        }
    }
    
    uint output_idx = c * output_height * output_width + 
                      out_h * output_width + out_w;
    output[output_idx] = max_val;
}

// 4. Fully Connected (Dense) Layer Kernel
kernel void fully_connected(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& input_size [[buffer(4)]],
    constant uint& output_size [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= output_size)
        return;
    
    float sum = bias[gid];
    
    for (uint i = 0; i < input_size; i++) {
        uint weight_idx = gid * input_size + i;
        sum += input[i] * weights[weight_idx];
    }
    
    output[gid] = sum;
}

// 5. Optimized Convolution with Threadgroup Memory
// NOTE: Simplified version - just use the same approach as basic for now
kernel void conv2d_3x3_optimized(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& input_height [[buffer(4)]],
    constant uint& input_width [[buffer(5)]],
    constant uint& input_channels [[buffer(6)]],
    constant uint& output_channels [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // Simplified: use same logic as basic conv2d_3x3
    // Proper tile-based optimization with threadgroup memory is complex
    // and requires careful handling of tile sizes, boundaries, etc.

    uint out_h = gid.x;
    uint out_w = gid.y;
    uint out_c = gid.z;

    if (out_h >= input_height || out_w >= input_width || out_c >= output_channels)
        return;

    float sum = bias[out_c];

    // 3x3 convolution with padding
    for (uint in_c = 0; in_c < input_channels; in_c++) {
        for (int kh = -1; kh <= 1; kh++) {
            for (int kw = -1; kw <= 1; kw++) {
                int in_h = int(out_h) + kh;
                int in_w = int(out_w) + kw;

                // Apply padding (zero outside bounds)
                if (in_h >= 0 && in_h < int(input_height) &&
                    in_w >= 0 && in_w < int(input_width)) {

                    uint input_idx = in_c * input_height * input_width +
                                    uint(in_h) * input_width + uint(in_w);

                    uint weight_idx = out_c * input_channels * 9 +
                                     in_c * 9 +
                                     uint(kh + 1) * 3 + uint(kw + 1);

                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }

    uint output_idx = out_c * input_height * input_width +
                      out_h * input_width + out_w;
    output[output_idx] = sum;
}

// 6. Batch Normalization Kernel
kernel void batch_norm(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device const float* mean [[buffer(3)]],
    device const float* variance [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& height [[buffer(6)]],
    constant uint& width [[buffer(7)]],
    constant uint& channels [[buffer(8)]],
    constant float& epsilon [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint h = gid.x;
    uint w = gid.y;
    uint c = gid.z;
    
    if (h >= height || w >= width || c >= channels)
        return;
    
    uint idx = c * height * width + h * width + w;
    
    float normalized = (input[idx] - mean[c]) / sqrt(variance[c] + epsilon);
    output[idx] = gamma[c] * normalized + beta[c];
}

// 7. Softmax Kernel
kernel void softmax(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size)
        return;
    
    // Find maximum for numerical stability
    float max_val = input[0];
    for (uint i = 1; i < size; i++) {
        max_val = max(max_val, input[i]);
    }
    
    // Calculate exponentials and sum
    float sum = 0.0f;
    for (uint i = 0; i < size; i++) {
        sum += exp(input[i] - max_val);
    }
    
    // Calculate softmax
    output[gid] = exp(input[gid] - max_val) / sum;
}

// 8. Matrix Multiplication using SIMD (for fully connected layers)
// Uses 1D dispatch with X/Y macros from helpers.mako
kernel void matrix_multiply_simd(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* C [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    // Use X/Y macros with height = M for column-major indexing
    uint height = M;
    uint row = Y(gid);  // Y(addr) = addr % height
    uint col = X(gid);  // X(addr) = addr / height

    if (row >= M || col >= N)
        return;

    float4 sum = float4(0.0);
    uint k;

    // Process 4 elements at a time using SIMD
    for (k = 0; k + 3 < K; k += 4) {
        float4 a_vec = float4(A[row * K + k], A[row * K + k + 1],
                             A[row * K + k + 2], A[row * K + k + 3]);
        float4 b_vec = float4(B[k * N + col], B[(k + 1) * N + col],
                             B[(k + 2) * N + col], B[(k + 3) * N + col]);
        sum += a_vec * b_vec;
    }

    float result = sum.x + sum.y + sum.z + sum.w;

    // Handle remaining elements
    for (; k < K; k++) {
        result += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = result + bias[col];
}