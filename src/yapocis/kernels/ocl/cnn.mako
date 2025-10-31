// Unified CNN OpenCL Kernels
// Combines VGG16 and modern CNN layers into a comprehensive deep learning toolkit

// =================================================================
// Convolutional Layers
// =================================================================

// 3x3 Convolution (VGG16, ResNet, most CNNs)
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

    // 3x3 convolution with padding
    for (uint in_c = 0; in_c < input_channels; in_c++) {
        for (uint kh = 0; kh < 3; kh++) {
            for (uint kw = 0; kw < 3; kw++) {
                int in_h = (int)out_h + (int)kh - 1;  // padding = 1
                int in_w = (int)out_w + (int)kw - 1;  // padding = 1

                // Check bounds (zero padding)
                if (in_h >= 0 && in_h < (int)input_height &&
                    in_w >= 0 && in_w < (int)input_width) {

                    uint input_idx = in_c * input_height * input_width +
                                    (uint)in_h * input_width + (uint)in_w;
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

    // 3x3 convolution with padding
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
                                     in_c * 9 +
                                     (uint)(kh + 1) * 3 + (uint)(kw + 1);

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
__kernel void conv2d_1x1(
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
__kernel void depthwise_conv2d_3x3(
    __global const float* input,
    __global const float* weights,
    __global const float* bias,
    __global float* output,
    const uint input_height,
    const uint input_width,
    const uint channels
) {
    const uint out_h = get_global_id(0);
    const uint out_w = get_global_id(1);
    const uint c = get_global_id(2);

    if (out_h >= input_height || out_w >= input_width || c >= channels)
        return;

    float sum = bias[c];

    // 3x3 depthwise convolution - only convolve within the same channel
    for (uint kh = 0; kh < 3; kh++) {
        for (uint kw = 0; kw < 3; kw++) {
            int in_h = (int)out_h + (int)kh - 1;  // padding = 1
            int in_w = (int)out_w + (int)kw - 1;  // padding = 1

            if (in_h >= 0 && in_h < (int)input_height &&
                in_w >= 0 && in_w < (int)input_width) {

                uint input_idx = c * input_height * input_width +
                                (uint)in_h * input_width + (uint)in_w;
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
__kernel void relu_activation(
    __global const float* input,
    __global float* output,
    const uint size
) {
    const uint gid = get_global_id(0);

    if (gid >= size)
        return;

    output[gid] = fmax(0.0f, input[gid]);
}

// GELU Activation (BERT, GPT, transformers)
__kernel void gelu(
    __global const float* input,
    __global float* output,
    const uint size
) {
    const uint gid = get_global_id(0);

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
        max_val = fmax(max_val, input[i]);
    }

    // Calculate exponentials and sum
    float sum = 0.0f;
    for (uint i = 0; i < size; i++) {
        sum += exp(input[i] - max_val);
    }

    // Calculate softmax
    output[gid] = exp(input[gid] - max_val) / sum;
}

// Sigmoid Activation (RetinaNet, binary classification)
__kernel void sigmoid(
    __global const float* input,
    __global float* output,
    const uint size
) {
    const uint gid = get_global_id(0);

    if (gid >= size)
        return;

    output[gid] = 1.0f / (1.0f + exp(-input[gid]));
}

// =================================================================
// Pooling Layers
// =================================================================

// 2x2 Max Pooling (VGG16, traditional CNNs)
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

            max_val = fmax(max_val, input[input_idx]);
        }
    }

    uint output_idx = c * output_height * output_width +
                      out_h * output_width + out_w;
    output[output_idx] = max_val;
}

// 3x3 Max Pooling with stride=2, padding=1 (ResNet)
__kernel void max_pool_3x3_stride2(
    __global const float* input,
    __global float* output,
    const uint input_height,
    const uint input_width,
    const uint channels
) {
    const uint out_h = get_global_id(0);
    const uint out_w = get_global_id(1);
    const uint c = get_global_id(2);

    uint output_height = input_height / 2;
    uint output_width = input_width / 2;

    if (out_h >= output_height || out_w >= output_width || c >= channels)
        return;

    float max_val = -INFINITY;

    // 3x3 pooling window with stride=2, padding=1
    for (uint kh = 0; kh < 3; kh++) {
        for (uint kw = 0; kw < 3; kw++) {
            int in_h = (int)(out_h * 2) - 1 + (int)kh;
            int in_w = (int)(out_w * 2) - 1 + (int)kw;

            // Check bounds (padding)
            if (in_h >= 0 && in_h < (int)input_height &&
                in_w >= 0 && in_w < (int)input_width) {
                uint input_idx = c * input_height * input_width +
                                (uint)in_h * input_width + (uint)in_w;
                max_val = fmax(max_val, input[input_idx]);
            }
        }
    }

    uint output_idx = c * output_height * output_width +
                      out_h * output_width + out_w;
    output[output_idx] = max_val;
}

// Global Average Pooling (ResNet, MobileNet)
__kernel void global_avg_pool(
    __global const float* input,
    __global float* output,
    const uint input_height,
    const uint input_width,
    const uint channels
) {
    const uint c = get_global_id(0);

    if (c >= channels)
        return;

    float sum = 0.0f;
    uint spatial_size = input_height * input_width;

    // Sum all spatial positions
    for (uint h = 0; h < input_height; h++) {
        for (uint w = 0; w < input_width; w++) {
            uint input_idx = c * spatial_size + h * input_width + w;
            sum += input[input_idx];
        }
    }

    // Average over spatial dimensions
    output[c] = sum / (float)spatial_size;
}

// Global Max Pooling
__kernel void global_max_pool(
    __global const float* input,
    __global float* output,
    const uint input_height,
    const uint input_width,
    const uint channels
) {
    const uint c = get_global_id(0);

    if (c >= channels)
        return;

    float max_val = -INFINITY;
    uint spatial_size = input_height * input_width;

    // Find max across all spatial positions
    for (uint h = 0; h < input_height; h++) {
        for (uint w = 0; w < input_width; w++) {
            uint input_idx = c * spatial_size + h * input_width + w;
            max_val = fmax(max_val, input[input_idx]);
        }
    }

    output[c] = max_val;
}

// =================================================================
// Normalization Layers
// =================================================================

// Batch Normalization
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

// Layer Normalization
__kernel void layer_norm(
    __global const float* input,
    __global const float* gamma,
    __global const float* beta,
    __global float* output,
    const uint input_height,
    const uint input_width,
    const uint channels,
    const float epsilon
) {
    const uint h = get_global_id(0);
    const uint w = get_global_id(1);

    if (h >= input_height || w >= input_width)
        return;

    // Compute mean across channels
    float mean = 0.0f;
    for (uint c = 0; c < channels; c++) {
        uint idx = c * input_height * input_width + h * input_width + w;
        mean += input[idx];
    }
    mean /= (float)channels;

    // Compute variance across channels
    float variance = 0.0f;
    for (uint c = 0; c < channels; c++) {
        uint idx = c * input_height * input_width + h * input_width + w;
        float diff = input[idx] - mean;
        variance += diff * diff;
    }
    variance /= (float)channels;

    // Normalize and apply scale/shift
    float std = sqrt(variance + epsilon);
    for (uint c = 0; c < channels; c++) {
        uint idx = c * input_height * input_width + h * input_width + w;
        float normalized = (input[idx] - mean) / std;
        output[idx] = gamma[c] * normalized + beta[c];
    }
}

// 1D Layer Normalization (for transformers/sequences)
__kernel void layer_norm_1d(
    __global const float* input,
    __global const float* gamma,
    __global const float* beta,
    __global float* output,
    const uint size,
    const float epsilon
) {
    const uint gid = get_global_id(0);

    if (gid > 0)
        return;

    // Compute mean
    float mean = 0.0f;
    for (uint i = 0; i < size; i++) {
        mean += input[i];
    }
    mean /= (float)size;

    // Compute variance
    float variance = 0.0f;
    for (uint i = 0; i < size; i++) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }
    variance /= (float)size;

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

// Matrix Multiplication with SIMD (simplified for OpenCL)
__kernel void matrix_multiply_simd(
    __global const float* A,
    __global const float* B,
    __global const float* bias,
    __global float* C,
    const uint M,
    const uint N,
    const uint K
) {
    const uint gid = get_global_id(0);

    uint row = gid % M;
    uint col = gid / M;

    if (row >= M || col >= N)
        return;

    float sum = 0.0f;

    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = sum + bias[col];
}

// =================================================================
// Regularization
// =================================================================

// Linear congruential generator for random numbers
inline uint lcg_random(uint seed) {
    return seed * 1664525u + 1013904223u;
}

// Dropout
__kernel void dropout(
    __global const float* input,
    __global float* output,
    const float rate,
    const uint seed,
    const uint training,
    const uint size
) {
    const uint gid = get_global_id(0);

    if (gid >= size)
        return;

    // Inference mode: identity
    if (training == 0) {
        output[gid] = input[gid];
        return;
    }

    // Training mode: randomly drop elements
    uint random_state = lcg_random(seed + gid);
    float random_val = (float)random_state / (float)UINT_MAX;

    if (random_val < rate) {
        output[gid] = 0.0f;
    } else {
        output[gid] = input[gid] / (1.0f - rate);
    }
}

// Spatial Dropout
__kernel void spatial_dropout(
    __global const float* input,
    __global float* output,
    const uint channels,
    const uint height,
    const uint width,
    const float rate,
    const uint seed,
    const uint training
) {
    const uint h = get_global_id(0);
    const uint w = get_global_id(1);
    const uint c = get_global_id(2);

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
    float random_val = (float)random_state / (float)UINT_MAX;

    if (random_val < rate) {
        output[idx] = 0.0f;
    } else {
        output[idx] = input[idx] / (1.0f - rate);
    }
}
// =================================================================
// ResNet-specific Convolution Kernels
// =================================================================

// 7x7 Convolution with stride=2, padding=3
__kernel void conv2d_7x7_stride2(
    __global const float* input,
    __global const float* weights,
    __global const float* bias,
    __global float* output,
    const uint input_height,
    const uint input_width,
    const uint input_channels,
    const uint output_channels
) {
    const uint out_h = get_global_id(0);  // height
    const uint out_c = get_global_id(1);  // channels
    const uint out_w = get_global_id(2);  // width

    const uint output_height = input_height / 2;
    const uint output_width = input_width / 2;

    if (out_h >= output_height || out_w >= output_width || out_c >= output_channels)
        return;

    float sum = bias[out_c];

    // Stride=2 means we sample every 2nd input pixel
    int in_h_start = (int)out_h * 2 - 3;  // padding=3
    int in_w_start = (int)out_w * 2 - 3;

    for (uint in_c = 0; in_c < input_channels; in_c++) {
        for (uint kh = 0; kh < 7; kh++) {
            for (uint kw = 0; kw < 7; kw++) {
                int in_h = in_h_start + (int)kh;
                int in_w = in_w_start + (int)kw;

                // Zero padding
                if (in_h >= 0 && in_h < (int)input_height &&
                    in_w >= 0 && in_w < (int)input_width) {
                    uint input_idx = in_c * input_height * input_width +
                                    (uint)in_h * input_width + (uint)in_w;
                    uint weight_idx = ((out_c * input_channels + in_c) * 7 + kh) * 7 + kw;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }

    uint output_idx = out_c * output_height * output_width + out_h * output_width + out_w;
    output[output_idx] = sum;
}

// 3x3 Convolution with stride=2, padding=1
__kernel void conv2d_3x3_stride2(
    __global const float* input,
    __global const float* weights,
    __global const float* bias,
    __global float* output,
    const uint input_height,
    const uint input_width,
    const uint input_channels,
    const uint output_channels
) {
    const uint out_h = get_global_id(0);  // height
    const uint out_c = get_global_id(1);  // channels
    const uint out_w = get_global_id(2);  // width

    // Output size with padding=1, kernel=3, stride=2: (input - 1) / 2 + 1
    const uint output_height = (input_height - 1) / 2 + 1;
    const uint output_width = (input_width - 1) / 2 + 1;

    if (out_h >= output_height || out_w >= output_width || out_c >= output_channels)
        return;

    float sum = bias[out_c];

    int in_h_start = (int)out_h * 2 - 1;  // padding=1
    int in_w_start = (int)out_w * 2 - 1;

    for (uint in_c = 0; in_c < input_channels; in_c++) {
        for (uint kh = 0; kh < 3; kh++) {
            for (uint kw = 0; kw < 3; kw++) {
                int in_h = in_h_start + (int)kh;
                int in_w = in_w_start + (int)kw;

                if (in_h >= 0 && in_h < (int)input_height &&
                    in_w >= 0 && in_w < (int)input_width) {
                    uint input_idx = in_c * input_height * input_width +
                                    (uint)in_h * input_width + (uint)in_w;
                    uint weight_idx = ((out_c * input_channels + in_c) * 3 + kh) * 3 + kw;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }

    uint output_idx = out_c * output_height * output_width + out_h * output_width + out_w;
    output[output_idx] = sum;
}

// 1x1 Convolution with stride=2
__kernel void conv2d_1x1_stride2(
    __global const float* input,
    __global const float* weights,
    __global const float* bias,
    __global float* output,
    const uint input_height,
    const uint input_width,
    const uint input_channels,
    const uint output_channels
) {
    const uint out_h = get_global_id(0);  // height
    const uint out_c = get_global_id(1);  // channels
    const uint out_w = get_global_id(2);  // width

    const uint output_height = input_height / 2;
    const uint output_width = input_width / 2;

    if (out_h >= output_height || out_w >= output_width || out_c >= output_channels)
        return;

    float sum = bias[out_c];

    // Stride=2: sample every 2nd pixel
    uint in_h = out_h * 2;
    uint in_w = out_w * 2;

    for (uint in_c = 0; in_c < input_channels; in_c++) {
        uint input_idx = in_c * input_height * input_width + in_h * input_width + in_w;
        uint weight_idx = out_c * input_channels + in_c;
        sum += input[input_idx] * weights[weight_idx];
    }

    uint output_idx = out_c * output_height * output_width + out_h * output_width + out_w;
    output[output_idx] = sum;
}

// =================================================================
// Element-wise Operations
// =================================================================

// Element-wise addition (for residual connections)
__kernel void add_tensors(
    __global const float* input1,
    __global const float* input2,
    __global float* output,
    const uint size
) {
    const uint gid = get_global_id(0);

    if (gid >= size) return;

    output[gid] = input1[gid] + input2[gid];
}

// =================================================================
// Upsampling / Spatial Operations
// =================================================================

// 2x Bilinear Upsampling (FPN, U-Net, semantic segmentation)
__kernel void upsample_2x(
    __global const float* input,
    __global float* output,
    const uint input_height,
    const uint input_width,
    const uint channels
) {
    // dispatch_size is (height_of output, depth_of output, width_of output)
    // which maps to (out_h, channels, out_w)
    const uint out_h = get_global_id(0);
    const uint c = get_global_id(1);
    const uint out_w = get_global_id(2);

    const uint output_height = input_height * 2;
    const uint output_width = input_width * 2;

    if (out_h >= output_height || c >= channels || out_w >= output_width)
        return;

    // Nearest neighbor interpolation (matches torchvision FPN)
    // Each input pixel maps to 4 output pixels in a 2×2 block
    uint in_h = out_h / 2;
    uint in_w = out_w / 2;

    // Read input pixel and replicate to output
    uint input_idx = c * input_height * input_width + in_h * input_width + in_w;
    uint output_idx = c * output_height * output_width + out_h * output_width + out_w;

    output[output_idx] = input[input_idx];
}
