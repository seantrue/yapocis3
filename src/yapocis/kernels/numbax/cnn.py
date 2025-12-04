"""
Unified CNN kernels for Numbax backend.

Implements all CNN layers: convolution, activation, pooling, normalization,
fully connected, and regularization for VGG16, ResNet, MobileNet, and Transformers.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def _wrap(f):
            return f

        return _wrap

    def prange(*args, **kwargs):  # type: ignore
        return range(*args)


# =================================================================
# Convolutional Layers
# =================================================================


@njit(cache=True, fastmath=True, parallel=True)
def _conv2d_3x3_njit(
    input: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    output: np.ndarray,
    output_channels: int,
) -> None:
    """3x3 convolution with padding=1, stride=1."""
    # Input: (C_in, H, W)
    # Weights: (C_out, C_in, 3, 3)
    # Output: (C_out, H, W)
    C_in, H, W = input.shape
    C_out = output_channels

    for c_out in prange(C_out):
        for y in range(H):
            for x in range(W):
                sum_val = bias[c_out]
                for c_in in range(C_in):
                    for ky in range(3):
                        for kx in range(3):
                            # Padding=1: coordinates with padding
                            py = y + ky - 1
                            px = x + kx - 1
                            # Zero padding
                            if 0 <= py < H and 0 <= px < W:
                                sum_val += (
                                    input[c_in, py, px] * weights[c_out, c_in, ky, kx]
                                )
                output[c_out, y, x] = sum_val


def conv2d_3x3(
    input: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    output: np.ndarray,
    input_height: int,
    input_depth: int,
    input_width: int,
    output_channels: int,
) -> np.ndarray:
    """3x3 convolution with padding=1, stride=1."""
    if HAVE_NUMBA:
        _conv2d_3x3_njit(input, weights, bias, output, output_channels)
    else:
        # NumPy fallback using correlate
        C_in, H, W = input.shape
        C_out = output_channels
        for c_out in range(C_out):
            output[c_out] = bias[c_out]
            for c_in in range(C_in):
                # Pad input
                padded = np.pad(input[c_in], ((1, 1), (1, 1)), mode="constant")
                # Convolve
                for y in range(H):
                    for x in range(W):
                        output[c_out, y, x] += np.sum(
                            padded[y : y + 3, x : x + 3] * weights[c_out, c_in, :, :]
                        )
    return output


def conv2d_3x3_res(
    input: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    output: np.ndarray,
    input_height: int,
    input_depth: int,
    input_width: int,
    output_channels: int,
) -> None:
    """Resident version of conv2d_3x3."""
    conv2d_3x3(
        input,
        weights,
        bias,
        output,
        input_height,
        input_depth,
        input_width,
        output_channels,
    )
    return None


# Optimized version is the same for Numbax (already optimized with parallel)
def conv2d_3x3_optimized(
    input: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    output: np.ndarray,
    input_height: int,
    input_depth: int,
    input_width: int,
    output_channels: int,
) -> np.ndarray:
    """Optimized 3x3 convolution (same as standard for Numbax)."""
    return conv2d_3x3(
        input,
        weights,
        bias,
        output,
        input_height,
        input_depth,
        input_width,
        output_channels,
    )


def conv2d_3x3_optimized_res(
    input: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    output: np.ndarray,
    input_height: int,
    input_depth: int,
    input_width: int,
    output_channels: int,
) -> None:
    """Resident version of conv2d_3x3_optimized."""
    conv2d_3x3_optimized(
        input,
        weights,
        bias,
        output,
        input_height,
        input_depth,
        input_width,
        output_channels,
    )
    return None


@njit(cache=True, fastmath=True, parallel=True)
def _conv2d_1x1_njit(
    input: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    output: np.ndarray,
    output_channels: int,
) -> None:
    """1x1 convolution for channel-wise transformations."""
    # Input: (C_in, H, W)
    # Weights: (C_out, C_in, 1, 1)
    # Output: (C_out, H, W)
    C_in, H, W = input.shape
    C_out = output_channels

    for c_out in prange(C_out):
        for y in range(H):
            for x in range(W):
                sum_val = bias[c_out]
                for c_in in range(C_in):
                    sum_val += input[c_in, y, x] * weights[c_out, c_in, 0, 0]
                output[c_out, y, x] = sum_val


def conv2d_1x1(
    input: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    output: np.ndarray,
    input_height: int,
    input_depth: int,
    input_width: int,
    output_channels: int,
) -> np.ndarray:
    """1x1 convolution for channel-wise transformations."""
    if HAVE_NUMBA:
        _conv2d_1x1_njit(input, weights, bias, output, output_channels)
    else:
        C_in, H, W = input.shape
        C_out = output_channels
        for c_out in range(C_out):
            output[c_out] = bias[c_out]
            for c_in in range(C_in):
                output[c_out] += input[c_in] * weights[c_out, c_in, 0, 0]
    return output


def conv2d_1x1_res(
    input: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    output: np.ndarray,
    input_height: int,
    input_depth: int,
    input_width: int,
    output_channels: int,
) -> None:
    """Resident version of conv2d_1x1."""
    conv2d_1x1(
        input,
        weights,
        bias,
        output,
        input_height,
        input_depth,
        input_width,
        output_channels,
    )
    return None


@njit(cache=True, fastmath=True, parallel=True)
def _depthwise_conv2d_3x3_njit(
    input: np.ndarray, weights: np.ndarray, bias: np.ndarray, output: np.ndarray
) -> None:
    """Depthwise 3x3 convolution - one filter per channel."""
    # Input: (C, H, W)
    # Weights: (C, 1, 3, 3)
    # Output: (C, H, W)
    C, H, W = input.shape

    for c in prange(C):
        for y in range(H):
            for x in range(W):
                sum_val = bias[c]
                for ky in range(3):
                    for kx in range(3):
                        py = y + ky - 1
                        px = x + kx - 1
                        if 0 <= py < H and 0 <= px < W:
                            sum_val += input[c, py, px] * weights[c, 0, ky, kx]
                output[c, y, x] = sum_val


def depthwise_conv2d_3x3(
    input: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    output: np.ndarray,
    input_height: int,
    input_depth: int,
    input_width: int,
) -> np.ndarray:
    """Depthwise 3x3 convolution - one filter per channel."""
    if HAVE_NUMBA:
        _depthwise_conv2d_3x3_njit(input, weights, bias, output)
    else:
        C, H, W = input.shape
        for c in range(C):
            output[c] = bias[c]
            padded = np.pad(input[c], ((1, 1), (1, 1)), mode="constant")
            for y in range(H):
                for x in range(W):
                    output[c, y, x] += np.sum(
                        padded[y : y + 3, x : x + 3] * weights[c, 0, :, :]
                    )
    return output


def depthwise_conv2d_3x3_res(
    input: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    output: np.ndarray,
    input_height: int,
    input_depth: int,
    input_width: int,
) -> None:
    """Resident version of depthwise_conv2d_3x3."""
    depthwise_conv2d_3x3(
        input, weights, bias, output, input_height, input_depth, input_width
    )
    return None


# =================================================================
# Activation Functions
# =================================================================


@njit(cache=True, fastmath=True, parallel=True)
def _relu_activation_njit(input: np.ndarray, output: np.ndarray) -> None:
    """ReLU activation: max(0, x)."""
    inp_flat = input.ravel()
    out_flat = output.ravel()
    for i in prange(inp_flat.shape[0]):
        out_flat[i] = max(0.0, inp_flat[i])


def relu_activation(input: np.ndarray, output: np.ndarray, size: int) -> np.ndarray:
    """ReLU activation: max(0, x)."""
    if HAVE_NUMBA:
        _relu_activation_njit(input, output)
    else:
        np.maximum(input, 0.0, out=output)
    return output


def relu_activation_res(input: np.ndarray, output: np.ndarray, size: int) -> None:
    """Resident version of relu_activation."""
    relu_activation(input, output, size)
    return None


@njit(cache=True, fastmath=True, parallel=True)
def _gelu_njit(input: np.ndarray, output: np.ndarray) -> None:
    """GELU activation."""
    inp_flat = input.ravel()
    out_flat = output.ravel()
    sqrt_2_over_pi = 0.7978845608

    for i in prange(inp_flat.shape[0]):
        x = inp_flat[i]
        x_cubed = x * x * x
        inner = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
        # tanh approximation
        tanh_val = np.tanh(inner)
        out_flat[i] = 0.5 * x * (1.0 + tanh_val)


def gelu(input: np.ndarray, output: np.ndarray, size: int) -> np.ndarray:
    """GELU activation."""
    if HAVE_NUMBA:
        _gelu_njit(input, output)
    else:
        x = input
        sqrt_2_over_pi = 0.7978845608
        x_cubed = x * x * x
        inner = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
        output[:] = 0.5 * x * (1.0 + np.tanh(inner))
    return output


def gelu_res(input: np.ndarray, output: np.ndarray, size: int) -> None:
    """Resident version of gelu."""
    gelu(input, output, size)
    return None


@njit(cache=True, fastmath=True)
def _softmax_njit(input: np.ndarray, output: np.ndarray) -> None:
    """Softmax activation."""
    inp_flat = input.ravel()
    out_flat = output.ravel()
    n = inp_flat.shape[0]

    # Find max for numerical stability
    max_val = inp_flat[0]
    for i in range(1, n):
        if inp_flat[i] > max_val:
            max_val = inp_flat[i]

    # Compute exp(x - max) and sum
    sum_exp = 0.0
    for i in range(n):
        exp_val = np.exp(inp_flat[i] - max_val)
        out_flat[i] = exp_val
        sum_exp += exp_val

    # Normalize
    for i in range(n):
        out_flat[i] /= sum_exp


def softmax(input: np.ndarray, output: np.ndarray, size: int) -> np.ndarray:
    """Softmax activation."""
    if HAVE_NUMBA:
        _softmax_njit(input, output)
    else:
        x = input.ravel()
        max_val = np.max(x)
        exp_x = np.exp(x - max_val)
        output[:] = (exp_x / np.sum(exp_x)).reshape(input.shape)
    return output


def softmax_res(input: np.ndarray, output: np.ndarray, size: int) -> None:
    """Resident version of softmax."""
    softmax(input, output, size)
    return None


@njit(cache=True, fastmath=True, parallel=True)
def _sigmoid_njit(input: np.ndarray, output: np.ndarray) -> None:
    """Sigmoid activation: 1 / (1 + exp(-x))."""
    inp_flat = input.ravel()
    out_flat = output.ravel()

    for i in prange(inp_flat.shape[0]):
        out_flat[i] = 1.0 / (1.0 + np.exp(-inp_flat[i]))


def sigmoid(input: np.ndarray, output: np.ndarray, size: int) -> np.ndarray:
    """Sigmoid activation."""
    if HAVE_NUMBA:
        _sigmoid_njit(input, output)
    else:
        np.divide(1.0, 1.0 + np.exp(-input), out=output)
    return output


def sigmoid_res(input: np.ndarray, output: np.ndarray, size: int) -> None:
    """Resident version of sigmoid."""
    sigmoid(input, output, size)
    return None


# =================================================================
# Pooling Layers
# =================================================================


@njit(cache=True, fastmath=True, parallel=True)
def _max_pool_2x2_njit(input: np.ndarray, output: np.ndarray) -> None:
    """2x2 max pooling with stride=2."""
    # Input: (C, H, W)
    # Output: (C, H/2, W/2)
    C, H, W = input.shape
    H_out = H // 2
    W_out = W // 2

    for c in prange(C):
        for y_out in range(H_out):
            for x_out in range(W_out):
                y_in = y_out * 2
                x_in = x_out * 2
                max_val = input[c, y_in, x_in]
                for dy in range(2):
                    for dx in range(2):
                        val = input[c, y_in + dy, x_in + dx]
                        if val > max_val:
                            max_val = val
                output[c, y_out, x_out] = max_val


def max_pool_2x2(
    input: np.ndarray,
    output: np.ndarray,
    input_height: int,
    input_depth: int,
    input_width: int,
) -> np.ndarray:
    """2x2 max pooling with stride=2."""
    if HAVE_NUMBA:
        _max_pool_2x2_njit(input, output)
    else:
        C, H, W = input.shape
        H_out = H // 2
        W_out = W // 2
        for c in range(C):
            for y_out in range(H_out):
                for x_out in range(W_out):
                    y_in = y_out * 2
                    x_in = x_out * 2
                    output[c, y_out, x_out] = np.max(
                        input[c, y_in : y_in + 2, x_in : x_in + 2]
                    )
    return output


def max_pool_2x2_res(
    input: np.ndarray,
    output: np.ndarray,
    input_height: int,
    input_depth: int,
    input_width: int,
) -> None:
    """Resident version of max_pool_2x2."""
    max_pool_2x2(input, output, input_height, input_depth, input_width)
    return None


@njit(cache=True, fastmath=True, parallel=True)
def _max_pool_3x3_stride2_njit(input: np.ndarray, output: np.ndarray) -> None:
    """3x3 max pooling with stride=2, padding=1 (ResNet)."""
    # Input: (C, H, W)
    # Output: (C, H/2, W/2)
    C, H, W = input.shape
    H_out = H // 2
    W_out = W // 2

    for c in prange(C):
        for y_out in range(H_out):
            for x_out in range(W_out):
                # Start position with padding=1, stride=2
                y_in_start = y_out * 2 - 1
                x_in_start = x_out * 2 - 1

                max_val = float("-inf")

                # 3x3 pooling window
                for dy in range(3):
                    for dx in range(3):
                        y_in = y_in_start + dy
                        x_in = x_in_start + dx

                        # Bounds checking (padding)
                        if 0 <= y_in < H and 0 <= x_in < W:
                            val = input[c, y_in, x_in]
                            if val > max_val:
                                max_val = val

                output[c, y_out, x_out] = max_val


def max_pool_3x3_stride2(
    input: np.ndarray,
    output: np.ndarray,
    input_height: int,
    input_depth: int,
    input_width: int,
) -> np.ndarray:
    """3x3 max pooling with stride=2, padding=1 (ResNet)."""
    if HAVE_NUMBA:
        _max_pool_3x3_stride2_njit(input, output)
    else:
        C, H, W = input.shape
        H_out = H // 2
        W_out = W // 2
        for c in range(C):
            for y_out in range(H_out):
                for x_out in range(W_out):
                    y_in_start = y_out * 2 - 1
                    x_in_start = x_out * 2 - 1
                    max_val = float("-inf")
                    for dy in range(3):
                        for dx in range(3):
                            y_in = y_in_start + dy
                            x_in = x_in_start + dx
                            if 0 <= y_in < H and 0 <= x_in < W:
                                max_val = max(max_val, input[c, y_in, x_in])
                    output[c, y_out, x_out] = max_val
    return output


def max_pool_3x3_stride2_res(
    input: np.ndarray,
    output: np.ndarray,
    input_height: int,
    input_depth: int,
    input_width: int,
) -> None:
    """Resident version of max_pool_3x3_stride2."""
    max_pool_3x3_stride2(input, output, input_height, input_depth, input_width)
    return None


@njit(cache=True, fastmath=True, parallel=True)
def _global_avg_pool_njit(input: np.ndarray, output: np.ndarray) -> None:
    """Global average pooling."""
    # Input: (C, H, W)
    # Output: (C,)
    C, H, W = input.shape
    total_pixels = H * W

    for c in prange(C):
        sum_val = 0.0
        for y in range(H):
            for x in range(W):
                sum_val += input[c, y, x]
        output[c] = sum_val / total_pixels


def global_avg_pool(
    input: np.ndarray,
    output: np.ndarray,
    input_height: int,
    input_depth: int,
    input_width: int,
) -> np.ndarray:
    """Global average pooling."""
    if HAVE_NUMBA:
        _global_avg_pool_njit(input, output)
    else:
        C, H, W = input.shape
        for c in range(C):
            output[c] = np.mean(input[c])
    return output


def global_avg_pool_res(
    input: np.ndarray,
    output: np.ndarray,
    input_height: int,
    input_depth: int,
    input_width: int,
) -> None:
    """Resident version of global_avg_pool."""
    global_avg_pool(input, output, input_height, input_depth, input_width)
    return None


@njit(cache=True, fastmath=True, parallel=True)
def _global_max_pool_njit(input: np.ndarray, output: np.ndarray) -> None:
    """Global max pooling."""
    # Input: (C, H, W)
    # Output: (C,)
    C, H, W = input.shape

    for c in prange(C):
        max_val = input[c, 0, 0]
        for y in range(H):
            for x in range(W):
                val = input[c, y, x]
                if val > max_val:
                    max_val = val
        output[c] = max_val


def global_max_pool(
    input: np.ndarray,
    output: np.ndarray,
    input_height: int,
    input_depth: int,
    input_width: int,
) -> np.ndarray:
    """Global max pooling."""
    if HAVE_NUMBA:
        _global_max_pool_njit(input, output)
    else:
        C, H, W = input.shape
        for c in range(C):
            output[c] = np.max(input[c])
    return output


def global_max_pool_res(
    input: np.ndarray,
    output: np.ndarray,
    input_height: int,
    input_depth: int,
    input_width: int,
) -> None:
    """Resident version of global_max_pool."""
    global_max_pool(input, output, input_height, input_depth, input_width)
    return None


# =================================================================
# Normalization Layers
# =================================================================


@njit(cache=True, fastmath=True, parallel=True)
def _batch_norm_njit(
    input: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    output: np.ndarray,
    epsilon: float,
) -> None:
    """Batch normalization."""
    # Input: (C, H, W)
    # Parameters: (C,) each
    C, H, W = input.shape

    for c in prange(C):
        scale = gamma[c] / np.sqrt(variance[c] + epsilon)
        shift = beta[c] - mean[c] * scale
        for y in range(H):
            for x in range(W):
                output[c, y, x] = input[c, y, x] * scale + shift


def batch_norm(
    input: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    output: np.ndarray,
    input_height: int,
    input_depth: int,
    input_width: int,
    epsilon: float,
) -> np.ndarray:
    """Batch normalization."""
    if HAVE_NUMBA:
        _batch_norm_njit(input, gamma, beta, mean, variance, output, epsilon)
    else:
        C, H, W = input.shape
        for c in range(C):
            scale = gamma[c] / np.sqrt(variance[c] + epsilon)
            shift = beta[c] - mean[c] * scale
            output[c] = input[c] * scale + shift
    return output


def batch_norm_res(
    input: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    output: np.ndarray,
    input_height: int,
    input_depth: int,
    input_width: int,
    epsilon: float,
) -> None:
    """Resident version of batch_norm."""
    batch_norm(
        input,
        gamma,
        beta,
        mean,
        variance,
        output,
        input_height,
        input_depth,
        input_width,
        epsilon,
    )
    return None


@njit(cache=True, fastmath=True, parallel=True)
def _layer_norm_njit(
    input: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    output: np.ndarray,
    epsilon: float,
) -> None:
    """Layer normalization for 2D/3D tensors."""
    # Input: (C, H, W)
    # Normalize across channels for each spatial position
    C, H, W = input.shape

    for y in prange(H):
        for x in range(W):
            # Compute mean across channels
            mean_val = 0.0
            for c in range(C):
                mean_val += input[c, y, x]
            mean_val /= C

            # Compute variance across channels
            var_val = 0.0
            for c in range(C):
                diff = input[c, y, x] - mean_val
                var_val += diff * diff
            var_val /= C

            # Normalize
            scale = 1.0 / np.sqrt(var_val + epsilon)
            for c in range(C):
                normalized = (input[c, y, x] - mean_val) * scale
                output[c, y, x] = gamma[c] * normalized + beta[c]


def layer_norm(
    input: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    output: np.ndarray,
    input_height: int,
    input_depth: int,
    input_width: int,
    epsilon: float,
) -> np.ndarray:
    """Layer normalization for 2D/3D tensors."""
    if HAVE_NUMBA:
        _layer_norm_njit(input, gamma, beta, output, epsilon)
    else:
        C, H, W = input.shape
        for y in range(H):
            for x in range(W):
                mean_val = np.mean([input[c, y, x] for c in range(C)])
                var_val = np.var([input[c, y, x] for c in range(C)])
                scale = 1.0 / np.sqrt(var_val + epsilon)
                for c in range(C):
                    normalized = (input[c, y, x] - mean_val) * scale
                    output[c, y, x] = gamma[c] * normalized + beta[c]
    return output


def layer_norm_res(
    input: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    output: np.ndarray,
    input_height: int,
    input_depth: int,
    input_width: int,
    epsilon: float,
) -> None:
    """Resident version of layer_norm."""
    layer_norm(
        input, gamma, beta, output, input_height, input_depth, input_width, epsilon
    )
    return None


@njit(cache=True, fastmath=True)
def _layer_norm_1d_njit(
    input: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    output: np.ndarray,
    epsilon: float,
) -> None:
    """1D layer normalization for transformers."""
    # Input: (features,)
    inp_flat = input.ravel()
    out_flat = output.ravel()
    n = inp_flat.shape[0]

    # Compute mean
    mean_val = 0.0
    for i in range(n):
        mean_val += inp_flat[i]
    mean_val /= n

    # Compute variance
    var_val = 0.0
    for i in range(n):
        diff = inp_flat[i] - mean_val
        var_val += diff * diff
    var_val /= n

    # Normalize
    scale = 1.0 / np.sqrt(var_val + epsilon)
    for i in range(n):
        normalized = (inp_flat[i] - mean_val) * scale
        out_flat[i] = gamma[i] * normalized + beta[i]


def layer_norm_1d(
    input: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    output: np.ndarray,
    size: int,
    epsilon: float,
) -> np.ndarray:
    """1D layer normalization for transformers."""
    if HAVE_NUMBA:
        _layer_norm_1d_njit(input, gamma, beta, output, epsilon)
    else:
        mean_val = np.mean(input)
        var_val = np.var(input)
        scale = 1.0 / np.sqrt(var_val + epsilon)
        normalized = (input - mean_val) * scale
        output[:] = gamma * normalized + beta
    return output


def layer_norm_1d_res(
    input: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    output: np.ndarray,
    size: int,
    epsilon: float,
) -> None:
    """Resident version of layer_norm_1d."""
    layer_norm_1d(input, gamma, beta, output, size, epsilon)
    return None


# =================================================================
# Fully Connected Layers
# =================================================================


@njit(cache=True, fastmath=True, parallel=True)
def _fully_connected_njit(
    input: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    output: np.ndarray,
    input_size: int,
    output_size: int,
) -> None:
    """Fully connected layer: y = Wx + b."""
    # Weights: (output_size, input_size)
    for i in prange(output_size):
        sum_val = bias[i]
        for j in range(input_size):
            sum_val += weights[i, j] * input[j]
        output[i] = sum_val


def fully_connected(
    input: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    output: np.ndarray,
    input_size: int,
    output_size: int,
) -> np.ndarray:
    """Fully connected layer: y = Wx + b."""
    if HAVE_NUMBA:
        _fully_connected_njit(input, weights, bias, output, input_size, output_size)
    else:
        output[:] = np.dot(weights, input) + bias
    return output


def fully_connected_res(
    input: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    output: np.ndarray,
    input_size: int,
    output_size: int,
) -> None:
    """Resident version of fully_connected."""
    fully_connected(input, weights, bias, output, input_size, output_size)
    return None


@njit(cache=True, fastmath=True, parallel=True)
def _matrix_multiply_simd_njit(
    A: np.ndarray,
    B: np.ndarray,
    bias: np.ndarray,
    C: np.ndarray,
    M: int,
    N: int,
    K: int,
) -> None:
    """Matrix multiplication with bias: C = A * B + bias."""
    # A: (M, K), B: (K, N), C: (M, N)
    for i in prange(M):
        for j in range(N):
            sum_val = bias[j]
            for k in range(K):
                sum_val += A[i, k] * B[k, j]
            C[i, j] = sum_val


def matrix_multiply_simd(
    A: np.ndarray,
    B: np.ndarray,
    bias: np.ndarray,
    C: np.ndarray,
    M: int,
    N: int,
    K: int,
) -> np.ndarray:
    """Matrix multiplication with bias: C = A * B + bias."""
    if HAVE_NUMBA:
        _matrix_multiply_simd_njit(A, B, bias, C, M, N, K)
    else:
        C[:] = np.dot(A, B) + bias
    return C


def matrix_multiply_simd_res(
    A: np.ndarray,
    B: np.ndarray,
    bias: np.ndarray,
    C: np.ndarray,
    M: int,
    N: int,
    K: int,
) -> None:
    """Resident version of matrix_multiply_simd."""
    matrix_multiply_simd(A, B, bias, C, M, N, K)
    return None


# =================================================================
# Regularization
# =================================================================


@njit(cache=True, fastmath=True, parallel=True)
def _dropout_njit(
    input: np.ndarray, output: np.ndarray, rate: float, seed: int, training: int
) -> None:
    """Dropout - randomly zero out elements during training."""
    inp_flat = input.ravel()
    out_flat = output.ravel()
    n = inp_flat.shape[0]

    if training == 0:
        # Inference mode: identity
        for i in prange(n):
            out_flat[i] = inp_flat[i]
    else:
        # Training mode: apply dropout
        scale = 1.0 / (1.0 - rate)
        # Simple LCG for random numbers
        for i in prange(n):
            # Each thread gets its own seed
            rng_state = (seed + i * 1103515245 + 12345) & 0x7FFFFFFF
            rand_val = (rng_state % 10000) / 10000.0
            if rand_val < rate:
                out_flat[i] = 0.0
            else:
                out_flat[i] = inp_flat[i] * scale


def dropout(
    input: np.ndarray,
    output: np.ndarray,
    rate: float,
    seed: int,
    training: int,
    size: int,
) -> np.ndarray:
    """Dropout - randomly zero out elements during training."""
    if HAVE_NUMBA:
        _dropout_njit(input, output, rate, seed, training)
    else:
        if training == 0:
            output[:] = input
        else:
            np.random.seed(seed)
            mask = np.random.random(input.shape) >= rate
            scale = 1.0 / (1.0 - rate)
            output[:] = input * mask * scale
    return output


def dropout_res(
    input: np.ndarray,
    output: np.ndarray,
    rate: float,
    seed: int,
    training: int,
    size: int,
) -> None:
    """Resident version of dropout."""
    dropout(input, output, rate, seed, training, size)
    return None


@njit(cache=True, fastmath=True, parallel=True)
def _spatial_dropout_njit(
    input: np.ndarray, output: np.ndarray, rate: float, seed: int, training: int
) -> None:
    """Spatial dropout - drops entire feature maps (channels)."""
    # Input: (C, H, W)
    C, H, W = input.shape

    if training == 0:
        # Inference mode: identity
        for c in prange(C):
            for y in range(H):
                for x in range(W):
                    output[c, y, x] = input[c, y, x]
    else:
        # Training mode: drop entire channels
        scale = 1.0 / (1.0 - rate)
        for c in prange(C):
            # Decide if this channel is dropped
            rng_state = (seed + c * 1103515245 + 12345) & 0x7FFFFFFF
            rand_val = (rng_state % 10000) / 10000.0

            if rand_val < rate:
                # Drop entire channel
                for y in range(H):
                    for x in range(W):
                        output[c, y, x] = 0.0
            else:
                # Keep channel (with scaling)
                for y in range(H):
                    for x in range(W):
                        output[c, y, x] = input[c, y, x] * scale


def spatial_dropout(
    input: np.ndarray,
    output: np.ndarray,
    width: int,
    height: int,
    depth: int,
    rate: float,
    seed: int,
    training: int,
) -> np.ndarray:
    """Spatial dropout - drops entire feature maps (channels)."""
    if HAVE_NUMBA:
        _spatial_dropout_njit(input, output, rate, seed, training)
    else:
        C, H, W = input.shape
        if training == 0:
            output[:] = input
        else:
            np.random.seed(seed)
            mask = np.random.random(C) >= rate
            scale = 1.0 / (1.0 - rate)
            for c in range(C):
                if mask[c]:
                    output[c] = input[c] * scale
                else:
                    output[c] = 0.0
    return output


def spatial_dropout_res(
    input: np.ndarray,
    output: np.ndarray,
    width: int,
    height: int,
    depth: int,
    rate: float,
    seed: int,
    training: int,
) -> None:
    """Resident version of spatial_dropout."""
    spatial_dropout(input, output, width, height, depth, rate, seed, training)
    return None


# =================================================================
# ResNet-specific Convolution Kernels
# =================================================================


@njit(cache=True, fastmath=True, parallel=True)
def _conv2d_7x7_stride2_njit(
    input: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    output: np.ndarray,
    output_channels: int,
) -> None:
    """7x7 convolution with padding=3, stride=2."""
    # Input: (C_in, H_in, W_in)
    # Weights: (C_out, C_in, 7, 7)
    # Output: (C_out, H_out, W_out) where H_out = H_in/2, W_out = W_in/2
    C_in, H_in, W_in = input.shape
    C_out = output_channels
    H_out = H_in // 2
    W_out = W_in // 2

    for c_out in prange(C_out):
        for y_out in range(H_out):
            for x_out in range(W_out):
                sum_val = bias[c_out]
                # Stride=2 means we sample every 2nd input pixel
                y_in_start = y_out * 2 - 3  # padding=3
                x_in_start = x_out * 2 - 3

                for c_in in range(C_in):
                    for ky in range(7):
                        for kx in range(7):
                            y_in = y_in_start + ky
                            x_in = x_in_start + kx
                            # Zero padding
                            if 0 <= y_in < H_in and 0 <= x_in < W_in:
                                sum_val += (
                                    input[c_in, y_in, x_in]
                                    * weights[c_out, c_in, ky, kx]
                                )
                output[c_out, y_out, x_out] = sum_val


def conv2d_7x7_stride2(
    input: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    output: np.ndarray,
    input_height: int,
    input_depth: int,
    input_width: int,
    output_channels: int,
) -> np.ndarray:
    """7x7 convolution with padding=3, stride=2."""
    if HAVE_NUMBA:
        _conv2d_7x7_stride2_njit(input, weights, bias, output, output_channels)
    else:
        # NumPy fallback
        C_in, H_in, W_in = input.shape
        C_out = output_channels
        H_out = H_in // 2
        W_out = W_in // 2
        for c_out in range(C_out):
            for y_out in range(H_out):
                for x_out in range(W_out):
                    sum_val = bias[c_out]
                    y_in_start = y_out * 2 - 3
                    x_in_start = x_out * 2 - 3
                    for c_in in range(C_in):
                        for ky in range(7):
                            for kx in range(7):
                                y_in = y_in_start + ky
                                x_in = x_in_start + kx
                                if 0 <= y_in < H_in and 0 <= x_in < W_in:
                                    sum_val += (
                                        input[c_in, y_in, x_in]
                                        * weights[c_out, c_in, ky, kx]
                                    )
                    output[c_out, y_out, x_out] = sum_val
    return output


def conv2d_7x7_stride2_res(
    input: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    output: np.ndarray,
    input_height: int,
    input_depth: int,
    input_width: int,
    output_channels: int,
) -> None:
    """Resident version of conv2d_7x7_stride2."""
    conv2d_7x7_stride2(
        input,
        weights,
        bias,
        output,
        input_height,
        input_depth,
        input_width,
        output_channels,
    )
    return None


@njit(cache=True, fastmath=True, parallel=True)
def _conv2d_3x3_stride2_njit(
    input: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    output: np.ndarray,
    output_channels: int,
) -> None:
    """3x3 convolution with padding=1, stride=2."""
    # Input: (C_in, H_in, W_in)
    # Weights: (C_out, C_in, 3, 3)
    # Output: (C_out, H_out, W_out) where H_out = H_in/2, W_out = W_in/2
    C_in, H_in, W_in = input.shape
    C_out = output_channels
    H_out = H_in // 2
    W_out = W_in // 2

    for c_out in prange(C_out):
        for y_out in range(H_out):
            for x_out in range(W_out):
                sum_val = bias[c_out]
                # Stride=2 means we sample every 2nd input pixel
                y_in_start = y_out * 2 - 1  # padding=1
                x_in_start = x_out * 2 - 1

                for c_in in range(C_in):
                    for ky in range(3):
                        for kx in range(3):
                            y_in = y_in_start + ky
                            x_in = x_in_start + kx
                            # Zero padding
                            if 0 <= y_in < H_in and 0 <= x_in < W_in:
                                sum_val += (
                                    input[c_in, y_in, x_in]
                                    * weights[c_out, c_in, ky, kx]
                                )
                output[c_out, y_out, x_out] = sum_val


def conv2d_3x3_stride2(
    input: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    output: np.ndarray,
    input_height: int,
    input_depth: int,
    input_width: int,
    output_channels: int,
) -> np.ndarray:
    """3x3 convolution with padding=1, stride=2."""
    if HAVE_NUMBA:
        _conv2d_3x3_stride2_njit(input, weights, bias, output, output_channels)
    else:
        # NumPy fallback
        C_in, H_in, W_in = input.shape
        C_out = output_channels
        H_out = H_in // 2
        W_out = W_in // 2
        for c_out in range(C_out):
            for y_out in range(H_out):
                for x_out in range(W_out):
                    sum_val = bias[c_out]
                    y_in_start = y_out * 2 - 1
                    x_in_start = x_out * 2 - 1
                    for c_in in range(C_in):
                        for ky in range(3):
                            for kx in range(3):
                                y_in = y_in_start + ky
                                x_in = x_in_start + kx
                                if 0 <= y_in < H_in and 0 <= x_in < W_in:
                                    sum_val += (
                                        input[c_in, y_in, x_in]
                                        * weights[c_out, c_in, ky, kx]
                                    )
                    output[c_out, y_out, x_out] = sum_val
    return output


def conv2d_3x3_stride2_res(
    input: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    output: np.ndarray,
    input_height: int,
    input_depth: int,
    input_width: int,
    output_channels: int,
) -> None:
    """Resident version of conv2d_3x3_stride2."""
    conv2d_3x3_stride2(
        input,
        weights,
        bias,
        output,
        input_height,
        input_depth,
        input_width,
        output_channels,
    )
    return None


@njit(cache=True, fastmath=True, parallel=True)
def _conv2d_1x1_stride2_njit(
    input: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    output: np.ndarray,
    output_channels: int,
) -> None:
    """1x1 convolution with stride=2."""
    # Input: (C_in, H_in, W_in)
    # Weights: (C_out, C_in, 1, 1)
    # Output: (C_out, H_out, W_out) where H_out = H_in/2, W_out = W_in/2
    C_in, H_in, W_in = input.shape
    C_out = output_channels
    H_out = H_in // 2
    W_out = W_in // 2

    for c_out in prange(C_out):
        for y_out in range(H_out):
            for x_out in range(W_out):
                sum_val = bias[c_out]
                # Stride=2: sample every 2nd pixel
                y_in = y_out * 2
                x_in = x_out * 2

                for c_in in range(C_in):
                    sum_val += input[c_in, y_in, x_in] * weights[c_out, c_in, 0, 0]
                output[c_out, y_out, x_out] = sum_val


def conv2d_1x1_stride2(
    input: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    output: np.ndarray,
    input_height: int,
    input_depth: int,
    input_width: int,
    output_channels: int,
) -> np.ndarray:
    """1x1 convolution with stride=2."""
    if HAVE_NUMBA:
        _conv2d_1x1_stride2_njit(input, weights, bias, output, output_channels)
    else:
        # NumPy fallback
        C_in, H_in, W_in = input.shape
        C_out = output_channels
        H_out = H_in // 2
        W_out = W_in // 2
        for c_out in range(C_out):
            for y_out in range(H_out):
                for x_out in range(W_out):
                    sum_val = bias[c_out]
                    y_in = y_out * 2
                    x_in = x_out * 2
                    for c_in in range(C_in):
                        sum_val += input[c_in, y_in, x_in] * weights[c_out, c_in, 0, 0]
                    output[c_out, y_out, x_out] = sum_val
    return output


def conv2d_1x1_stride2_res(
    input: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    output: np.ndarray,
    input_height: int,
    input_depth: int,
    input_width: int,
    output_channels: int,
) -> None:
    """Resident version of conv2d_1x1_stride2."""
    conv2d_1x1_stride2(
        input,
        weights,
        bias,
        output,
        input_height,
        input_depth,
        input_width,
        output_channels,
    )
    return None


# =================================================================
# Element-wise Operations
# =================================================================


@njit(cache=True, fastmath=True, parallel=True)
def _add_tensors_njit(
    input1: np.ndarray, input2: np.ndarray, output: np.ndarray, size: int
) -> None:
    """Element-wise addition for residual connections."""
    for i in prange(size):
        output.flat[i] = input1.flat[i] + input2.flat[i]


def add_tensors(
    input1: np.ndarray, input2: np.ndarray, output: np.ndarray, size: int
) -> np.ndarray:
    """Element-wise addition (for residual connections)."""
    if HAVE_NUMBA:
        _add_tensors_njit(input1, input2, output, size)
    else:
        # NumPy fallback
        np.add(input1, input2, out=output)
    return output


def add_tensors_res(
    input1: np.ndarray, input2: np.ndarray, output: np.ndarray, size: int
) -> None:
    """Resident version of add_tensors."""
    add_tensors(input1, input2, output, size)
    return None


# =================================================================
# Upsampling / Spatial Operations
# =================================================================


@njit(cache=True, fastmath=True, parallel=True)
def _upsample_2x_njit(input: np.ndarray, output: np.ndarray) -> None:
    """2x bilinear upsampling."""
    # Input: (C, H, W)
    # Output: (C, 2*H, 2*W)
    C, H_in, W_in = input.shape
    H_out = H_in * 2
    W_out = W_in * 2

    for c in prange(C):
        for y_out in range(H_out):
            for x_out in range(W_out):
                # Map output position to input space
                in_y = (y_out + 0.5) / 2.0 - 0.5
                in_x = (x_out + 0.5) / 2.0 - 0.5

                # Get integer coordinates
                y0 = int(np.floor(in_y))
                x0 = int(np.floor(in_x))
                y1 = y0 + 1
                x1 = x0 + 1

                # Clamp to valid range
                y0 = max(0, min(y0, H_in - 1))
                y1 = max(0, min(y1, H_in - 1))
                x0 = max(0, min(x0, W_in - 1))
                x1 = max(0, min(x1, W_in - 1))

                # Get interpolation weights
                y_weight = in_y - np.floor(in_y)
                x_weight = in_x - np.floor(in_x)

                # Read four neighboring pixels
                val00 = input[c, y0, x0]
                val01 = input[c, y0, x1]
                val10 = input[c, y1, x0]
                val11 = input[c, y1, x1]

                # Bilinear interpolation
                val0 = val00 * (1.0 - x_weight) + val01 * x_weight
                val1 = val10 * (1.0 - x_weight) + val11 * x_weight
                result = val0 * (1.0 - y_weight) + val1 * y_weight

                output[c, y_out, x_out] = result


def upsample_2x(
    input: np.ndarray,
    output: np.ndarray,
    input_height: int,
    input_depth: int,
    input_width: int,
) -> np.ndarray:
    """2x bilinear upsampling (FPN, U-Net, semantic segmentation)."""
    if HAVE_NUMBA:
        _upsample_2x_njit(input, output)
    else:
        # NumPy fallback using scipy or manual implementation
        from scipy import ndimage

        C, H, W = input.shape
        for c in range(C):
            output[c] = ndimage.zoom(input[c], 2.0, order=1)
    return output


def upsample_2x_res(
    input: np.ndarray,
    output: np.ndarray,
    input_height: int,
    input_depth: int,
    input_width: int,
) -> None:
    """Resident version of upsample_2x."""
    upsample_2x(input, output, input_height, input_depth, input_width)
    return None
