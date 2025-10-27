// Unified CNN Metal Compute Shaders
// Combines VGG16 and modern CNN layers into a comprehensive deep learning toolkit

<%include file="constants.mako" />
<%include file="helpers.mako" />

#include <metal_stdlib>
using namespace metal;

// =================================================================
// Convolutional Layers
// =================================================================

// 3x3 Convolution (VGG16, ResNet, most CNNs)
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

// Optimized 3x3 Convolution (simplified version)
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

// 1x1 Convolution (ResNet, MobileNet)
kernel void conv2d_1x1(
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

    // 1x1 convolution is just a weighted sum across input channels
    for (uint in_c = 0; in_c < input_channels; in_c++) {
        uint input_idx = in_c * input_height * input_width +
                        out_h * input_width + out_w;
        uint weight_idx = out_c * input_channels + in_c;

        sum += input[input_idx] * weights[weight_idx];
    }

    uint output_idx = out_c * input_height * input_width +
                      out_h * input_width + out_w;
    output[output_idx] = sum;
}

// Depthwise 3x3 Convolution (MobileNet, EfficientNet)
kernel void depthwise_conv2d_3x3(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& input_height [[buffer(4)]],
    constant uint& input_width [[buffer(5)]],
    constant uint& channels [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint out_h = gid.x;
    uint out_w = gid.y;
    uint c = gid.z;

    if (out_h >= input_height || out_w >= input_width || c >= channels)
        return;

    float sum = bias[c];

    // 3x3 depthwise convolution - only convolve within the same channel
    for (uint kh = 0; kh < 3; kh++) {
        for (uint kw = 0; kw < 3; kw++) {
            int in_h = int(out_h + kh) - 1;  // padding = 1
            int in_w = int(out_w + kw) - 1;  // padding = 1

            if (in_h >= 0 && in_h < int(input_height) &&
                in_w >= 0 && in_w < int(input_width)) {

                uint input_idx = c * input_height * input_width +
                                uint(in_h) * input_width + uint(in_w);
                uint weight_idx = c * 9 + kh * 3 + kw;

                sum += input[input_idx] * weights[weight_idx];
            }
        }
    }

    uint output_idx = c * input_height * input_width +
                      out_h * input_width + out_w;
    output[output_idx] = sum;
}

// =================================================================
// Activation Functions
// =================================================================

// ReLU Activation
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

// GELU Activation (BERT, GPT, transformers)
kernel void gelu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size)
        return;

    float x = input[gid];

    // GELU approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608f;
    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
    float gelu_val = 0.5f * x * (1.0f + tanh(inner));

    output[gid] = gelu_val;
}

// Softmax Activation
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

// =================================================================
// Pooling Layers
// =================================================================

// 2x2 Max Pooling (VGG16, traditional CNNs)
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

// 3x3 Max Pooling with stride=2, padding=1 (ResNet)
kernel void max_pool_3x3_stride2(
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

    // 3x3 pooling window with stride=2, padding=1
    // Start position: (out_h * 2 - 1, out_w * 2 - 1)
    for (uint kh = 0; kh < 3; kh++) {
        for (uint kw = 0; kw < 3; kw++) {
            int in_h = int(out_h * 2) - 1 + int(kh);
            int in_w = int(out_w * 2) - 1 + int(kw);

            // Check bounds (padding)
            if (in_h >= 0 && in_h < int(input_height) &&
                in_w >= 0 && in_w < int(input_width)) {
                uint input_idx = c * input_height * input_width +
                                uint(in_h) * input_width + uint(in_w);
                max_val = max(max_val, input[input_idx]);
            }
        }
    }

    uint output_idx = c * output_height * output_width +
                      out_h * output_width + out_w;
    output[output_idx] = max_val;
}

// Global Average Pooling (ResNet, MobileNet)
kernel void global_avg_pool(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& input_height [[buffer(2)]],
    constant uint& input_width [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= channels)
        return;

    uint c = gid;
    float sum = 0.0f;
    uint spatial_size = input_height * input_width;

    // Sum all spatial positions for this channel
    for (uint h = 0; h < input_height; h++) {
        for (uint w = 0; w < input_width; w++) {
            uint input_idx = c * spatial_size + h * input_width + w;
            sum += input[input_idx];
        }
    }

    // Average over spatial dimensions
    output[c] = sum / float(spatial_size);
}

// Global Max Pooling
kernel void global_max_pool(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& input_height [[buffer(2)]],
    constant uint& input_width [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= channels)
        return;

    uint c = gid;
    float max_val = -INFINITY;
    uint spatial_size = input_height * input_width;

    // Find max across all spatial positions
    for (uint h = 0; h < input_height; h++) {
        for (uint w = 0; w < input_width; w++) {
            uint input_idx = c * spatial_size + h * input_width + w;
            max_val = max(max_val, input[input_idx]);
        }
    }

    output[c] = max_val;
}

// =================================================================
// Normalization Layers
// =================================================================

// Batch Normalization
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

// Layer Normalization
kernel void layer_norm(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& input_height [[buffer(4)]],
    constant uint& input_width [[buffer(5)]],
    constant uint& channels [[buffer(6)]],
    constant float& epsilon [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint h = gid.x;
    uint w = gid.y;

    if (h >= input_height || w >= input_width)
        return;

    // Compute mean across channels
    float mean = 0.0f;
    for (uint c = 0; c < channels; c++) {
        uint idx = c * input_height * input_width + h * input_width + w;
        mean += input[idx];
    }
    mean /= float(channels);

    // Compute variance across channels
    float variance = 0.0f;
    for (uint c = 0; c < channels; c++) {
        uint idx = c * input_height * input_width + h * input_width + w;
        float diff = input[idx] - mean;
        variance += diff * diff;
    }
    variance /= float(channels);

    // Normalize and apply learned scale/shift
    float std = sqrt(variance + epsilon);
    for (uint c = 0; c < channels; c++) {
        uint idx = c * input_height * input_width + h * input_width + w;
        float normalized = (input[idx] - mean) / std;
        output[idx] = gamma[c] * normalized + beta[c];
    }
}

// 1D Layer Normalization (for transformers/sequences)
kernel void layer_norm_1d(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    constant float& epsilon [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid > 0)
        return;

    // Compute mean
    float mean = 0.0f;
    for (uint i = 0; i < size; i++) {
        mean += input[i];
    }
    mean /= float(size);

    // Compute variance
    float variance = 0.0f;
    for (uint i = 0; i < size; i++) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }
    variance /= float(size);

    // Normalize and apply scale/shift
    float std = sqrt(variance + epsilon);
    for (uint i = 0; i < size; i++) {
        float normalized = (input[i] - mean) / std;
        output[i] = gamma[i] * normalized + beta[i];
    }
}

// =================================================================
// Fully Connected Layers
// =================================================================

// Fully Connected (Dense) Layer
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

// Matrix Multiplication with SIMD
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
    uint height = M;
    uint row = Y(gid);
    uint col = X(gid);

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

// =================================================================
// Regularization
// =================================================================

// Linear congruential generator for random numbers
inline uint lcg_random(uint seed) {
    return seed * 1664525u + 1013904223u;
}

// Dropout
kernel void dropout(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& rate [[buffer(2)]],
    constant uint& seed [[buffer(3)]],
    constant uint& training [[buffer(4)]],
    constant uint& size [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size)
        return;

    // Inference mode: identity
    if (training == 0) {
        output[gid] = input[gid];
        return;
    }

    // Training mode: randomly drop elements
    uint random_state = lcg_random(seed + gid);
    float random_val = float(random_state) / float(UINT_MAX);

    if (random_val < rate) {
        output[gid] = 0.0f;
    } else {
        output[gid] = input[gid] / (1.0f - rate);
    }
}

// Spatial Dropout
kernel void spatial_dropout(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    constant uint& height [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant float& rate [[buffer(5)]],
    constant uint& seed [[buffer(6)]],
    constant uint& training [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint h = gid.x;
    uint w = gid.y;
    uint c = gid.z;

    if (h >= height || w >= width || c >= channels)
        return;

    uint idx = c * height * width + h * width + w;

    // Inference mode: identity
    if (training == 0) {
        output[idx] = input[idx];
        return;
    }

    // Generate random value per channel
    uint random_state = lcg_random(seed + c);
    float random_val = float(random_state) / float(UINT_MAX);

    if (random_val < rate) {
        output[idx] = 0.0f;
    } else {
        output[idx] = input[idx] / (1.0f - rate);
    }
}
// =================================================================
// =================================================================
// ResNet-specific Convolution Kernels
// =================================================================

// 7x7 Convolution with stride=2, padding=3
kernel void conv2d_7x7_stride2(
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
    uint out_y = gid.x;
    uint out_c = gid.y;
    uint out_x = gid.z;

    uint output_height = input_height / 2;
    uint output_width = input_width / 2;

    if (out_y >= output_height || out_c >= output_channels || out_x >= output_width) {
        return;
    }

    float sum = bias[out_c];

    // Stride=2 means we sample every 2nd input pixel
    int in_y_start = int(out_y) * 2 - 3;  // padding=3
    int in_x_start = int(out_x) * 2 - 3;

    for (uint c_in = 0; c_in < input_channels; c_in++) {
        for (uint ky = 0; ky < 7; ky++) {
            for (uint kx = 0; kx < 7; kx++) {
                int in_y = in_y_start + int(ky);
                int in_x = in_x_start + int(kx);

                // Zero padding
                if (in_y >= 0 && in_y < int(input_height) &&
                    in_x >= 0 && in_x < int(input_width)) {
                    uint input_idx = c_in * input_height * input_width +
                                    uint(in_y) * input_width + uint(in_x);
                    uint weight_idx = ((out_c * input_channels + c_in) * 7 + ky) * 7 + kx;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }

    uint output_idx = out_c * output_height * output_width + out_y * output_width + out_x;
    output[output_idx] = sum;
}

// 3x3 Convolution with stride=2, padding=1
kernel void conv2d_3x3_stride2(
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
    uint out_y = gid.x;
    uint out_c = gid.y;
    uint out_x = gid.z;

    uint output_height = input_height / 2;
    uint output_width = input_width / 2;

    if (out_y >= output_height || out_c >= output_channels || out_x >= output_width) {
        return;
    }

    float sum = bias[out_c];

    int in_y_start = int(out_y) * 2 - 1;  // padding=1
    int in_x_start = int(out_x) * 2 - 1;

    for (uint c_in = 0; c_in < input_channels; c_in++) {
        for (uint ky = 0; ky < 3; ky++) {
            for (uint kx = 0; kx < 3; kx++) {
                int in_y = in_y_start + int(ky);
                int in_x = in_x_start + int(kx);

                if (in_y >= 0 && in_y < int(input_height) &&
                    in_x >= 0 && in_x < int(input_width)) {
                    uint input_idx = c_in * input_height * input_width +
                                    uint(in_y) * input_width + uint(in_x);
                    uint weight_idx = ((out_c * input_channels + c_in) * 3 + ky) * 3 + kx;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }

    uint output_idx = out_c * output_height * output_width + out_y * output_width + out_x;
    output[output_idx] = sum;
}

// 1x1 Convolution with stride=2
kernel void conv2d_1x1_stride2(
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
    uint out_y = gid.x;
    uint out_c = gid.y;
    uint out_x = gid.z;

    uint output_height = input_height / 2;
    uint output_width = input_width / 2;

    if (out_y >= output_height || out_c >= output_channels || out_x >= output_width) {
        return;
    }

    float sum = bias[out_c];

    // Stride=2: sample every 2nd pixel
    uint in_y = out_y * 2;
    uint in_x = out_x * 2;

    for (uint c_in = 0; c_in < input_channels; c_in++) {
        uint input_idx = c_in * input_height * input_width + in_y * input_width + in_x;
        uint weight_idx = out_c * input_channels + c_in;
        sum += input[input_idx] * weights[weight_idx];
    }

    uint output_idx = out_c * output_height * output_width + out_y * output_width + out_x;
    output[output_idx] = sum;
}

// =================================================================
// Element-wise Operations
// =================================================================

// Element-wise addition (for residual connections)
kernel void add_tensors(
    device const float* input1 [[buffer(0)]],
    device const float* input2 [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    output[gid] = input1[gid] + input2[gid];
}
