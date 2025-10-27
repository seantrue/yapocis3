// VGG16 OpenCL Kernels

<%include file="constants.mako" />
<%include file="helpers.mako" />

// =================================================================
// Convolution Kernels
// =================================================================

// 1. 3x3 Convolution (padding=1, stride=1)
__kernel void conv2d_3x3(
    __global const float* input,
    __global const float* weights,
    __global const float* bias,
    __global float* output,
    const uint input_height,
    const uint input_width,
    const uint input_channels,
    const uint output_channels
) {
    const uint out_h = get_global_id(0);
    const uint out_w = get_global_id(1);
    const uint out_c = get_global_id(2);

    if (out_h >= input_height || out_w >= input_width || out_c >= output_channels)
        return;

    float sum = bias[out_c];

    // 3x3 convolution with padding=1
    for (uint in_c = 0; in_c < input_channels; in_c++) {
        for (int kh = -1; kh <= 1; kh++) {
            for (int kw = -1; kw <= 1; kw++) {
                int in_h = (int)out_h + kh;
                int in_w = (int)out_w + kw;

                // Zero padding outside bounds
                if (in_h >= 0 && in_h < (int)input_height &&
                    in_w >= 0 && in_w < (int)input_width) {

                    uint input_idx = in_c * input_height * input_width +
                                    (uint)in_h * input_width + (uint)in_w;
                    uint weight_idx = out_c * input_channels * 9 +
                                     in_c * 9 + (uint)(kh + 1) * 3 + (uint)(kw + 1);

                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }

    uint output_idx = out_c * input_height * input_width +
                      out_h * input_width + out_w;
    output[output_idx] = sum;
}

// 2. Optimized 3x3 Convolution (same as basic for now)
__kernel void conv2d_3x3_optimized(
    __global const float* input,
    __global const float* weights,
    __global const float* bias,
    __global float* output,
    const uint input_height,
    const uint input_width,
    const uint input_channels,
    const uint output_channels
) {
    const uint out_h = get_global_id(0);
    const uint out_w = get_global_id(1);
    const uint out_c = get_global_id(2);

    if (out_h >= input_height || out_w >= input_width || out_c >= output_channels)
        return;

    float sum = bias[out_c];

    for (uint in_c = 0; in_c < input_channels; in_c++) {
        for (int kh = -1; kh <= 1; kh++) {
            for (int kw = -1; kw <= 1; kw++) {
                int in_h = (int)out_h + kh;
                int in_w = (int)out_w + kw;

                if (in_h >= 0 && in_h < (int)input_height &&
                    in_w >= 0 && in_w < (int)input_width) {

                    uint input_idx = in_c * input_height * input_width +
                                    (uint)in_h * input_width + (uint)in_w;
                    uint weight_idx = out_c * input_channels * 9 +
                                     in_c * 9 + (uint)(kh + 1) * 3 + (uint)(kw + 1);

                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }

    uint output_idx = out_c * input_height * input_width +
                      out_h * input_width + out_w;
    output[output_idx] = sum;
}

// =================================================================
// Activation Functions
// =================================================================

// 3. ReLU Activation
__kernel void relu_activation(
    __global const float* input,
    __global float* output,
    const uint size
) {
    const uint gid = get_global_id(0);

    if (gid >= size)
        return;

    output[gid] = max(0.0f, input[gid]);
}

// 4. Softmax Activation
__kernel void softmax(
    __global const float* input,
    __global float* output,
    const uint size
) {
    const uint gid = get_global_id(0);

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

// 5. Max Pooling 2x2 (stride=2)
__kernel void max_pool_2x2(
    __global const float* input,
    __global float* output,
    const uint input_height,
    const uint input_width,
    const uint channels
) {
    const uint out_h = get_global_id(0);
    const uint out_w = get_global_id(1);
    const uint c = get_global_id(2);

    const uint output_height = input_height / 2;
    const uint output_width = input_width / 2;

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

// =================================================================
// Normalization Layers
// =================================================================

// 6. Batch Normalization
__kernel void batch_norm(
    __global const float* input,
    __global const float* gamma,
    __global const float* beta,
    __global const float* mean,
    __global const float* variance,
    __global float* output,
    const uint height,
    const uint width,
    const uint channels,
    const float epsilon
) {
    const uint h = get_global_id(0);
    const uint w = get_global_id(1);
    const uint c = get_global_id(2);

    if (h >= height || w >= width || c >= channels)
        return;

    uint idx = c * height * width + h * width + w;

    float normalized = (input[idx] - mean[c]) / sqrt(variance[c] + epsilon);
    output[idx] = gamma[c] * normalized + beta[c];
}

// =================================================================
// Fully Connected Layers
// =================================================================

// 7. Fully Connected (Dense) Layer
__kernel void fully_connected(
    __global const float* input,
    __global const float* weights,
    __global const float* bias,
    __global float* output,
    const uint input_size,
    const uint output_size
) {
    const uint gid = get_global_id(0);

    if (gid >= output_size)
        return;

    float sum = bias[gid];

    for (uint i = 0; i < input_size; i++) {
        uint weight_idx = gid * input_size + i;
        sum += input[i] * weights[weight_idx];
    }

    output[gid] = sum;
}

// 8. Matrix Multiplication (for FC layers)
__kernel void matrix_multiply_simd(
    __global const float* A,
    __global const float* B,
    __global const float* bias,
    __global float* C,
    const uint M,
    const uint N,
    const uint K
) {
    const uint row = get_global_id(0);
    const uint col = get_global_id(1);

    if (row >= M || col >= N)
        return;

    float sum = 0.0f;

    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = sum + bias[col];
}
