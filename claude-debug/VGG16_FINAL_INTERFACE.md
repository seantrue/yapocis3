# VGG16 Final Idiomatic Interface

## Summary

Created an idiomatic `.interface` file for VGG16 Metal kernels following yapocis conventions, using parameter hints like `out_like`, `sizeof` where appropriate instead of explicit scalar parameters.

**Status**: ✅ Complete
**File**: `src/yapocis/interfaces/vgg16.interface`
**Tests**: `tests/test_vgg16_kernels.py` (23 functional tests)

---

## Final Interface Design

### Key Design Principles

✅ **Use `out_like` for same-shape outputs**
- Activation functions (ReLU, Softmax) preserve shape → use `out_like input`

✅ **Use `sizeof` for buffer sizes**
- Fully connected layers need input/output sizes → use `sizeof input`, `sizeof output`

✅ **Use explicit `int` for tensor dimensions**
- Conv, pool, batch norm have complex multi-dimensional shapes → explicit dimensions required
- Cannot infer output shape from input alone (e.g., pooling halves dimensions)

✅ **Buffers before scalars**
- All buffer parameters (`in`, `out`, `resident`) come before scalar parameters (`int`, `float`)

---

## Interface Categories

### 1. Activation Functions (Simple - Use `out_like`)

```idl
# ReLU activation: max(0, x)
kernel relu_activation(in float input, out_like input);
extension res as relu_activation(resident float input, resident float output);

# Softmax activation for classification
kernel softmax(in float input, out_like input);
extension res as softmax(resident float input, resident float output);
```

**Why `out_like`?**
- Output has same shape as input
- Size can be automatically inferred
- Idiomatic yapocis style

**Usage**:
```python
input_data = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
output = prog.relu_activation(input_data)  # No size parameter needed!
```

---

### 2. Fully Connected Layers (Use `sizeof`)

```idl
# Fully connected layer: y = Wx + b
kernel fully_connected(in float input, in float weights, in float bias, out float output,
                      sizeof input, sizeof output);
extension res as fully_connected(resident float input, resident float weights,
                                 resident float bias, resident float output,
                                 sizeof input, sizeof output);
```

**Why `sizeof`?**
- Output size differs from input size
- Cannot use `out_like` as shapes don't match
- `sizeof` provides buffer sizes explicitly

**Usage**:
```python
input_data = np.random.randn(10).astype(np.float32)
weights = np.random.randn(5, 10).astype(np.float32)  # 5 outputs, 10 inputs
bias = np.zeros(5, dtype=np.float32)

# Output buffer is allocated automatically
output = prog.fully_connected(input_data, weights, bias, output_array)
# sizeof input = 10, sizeof output = 5 passed automatically
```

---

### 3. Convolutional Layers (Explicit dimensions)

```idl
# 3x3 Convolution with padding=1, stride=1
kernel conv2d_3x3(in float input, in float weights, in float bias, out float output,
                  int input_height, int input_width,
                  int input_channels, int output_channels);
extension res as conv2d_3x3(resident float input, resident float weights,
                           resident float bias, resident float output,
                           int input_height, int input_width,
                           int input_channels, int output_channels);
```

**Why explicit dimensions?**
- 3D tensors with shape (channels, height, width)
- Cannot infer relationships from buffer size alone
- Need to know which dimension is which

**Usage**:
```python
input_img = np.random.randn(3, 224, 224).astype(np.float32)  # RGB image
weights = np.random.randn(64, 3, 3, 3).astype(np.float32)    # 64 filters
bias = np.zeros(64, dtype=np.float32)

output = prog.conv2d_3x3(
    input_img, weights, bias, output_buffer,
    input_height=224,
    input_width=224,
    input_channels=3,
    output_channels=64
)
# Output shape: (64, 224, 224) - same spatial size due to padding
```

---

### 4. Pooling Layers (Explicit dimensions - shape changes)

```idl
# 2x2 Max Pooling with stride=2
kernel max_pool_2x2(in float input, out float output,
                    int input_height, int input_width, int channels);
extension res as max_pool_2x2(resident float input, resident float output,
                               int input_height, int input_width, int channels);
```

**Why explicit dimensions?**
- Output shape is DIFFERENT from input (halves spatial dimensions)
- Input: (channels, H, W) → Output: (channels, H/2, W/2)
- Must specify dimensions explicitly

**Usage**:
```python
input_data = np.random.randn(64, 224, 224).astype(np.float32)

# Output buffer must be pre-allocated with correct size
output_data = np.zeros((64, 112, 112), dtype=np.float32)

output = prog.max_pool_2x2(
    input_data, output_data,
    input_height=224,
    input_width=224,
    channels=64
)
# Output shape: (64, 112, 112) - half the spatial dimensions
```

---

### 5. Batch Normalization (Use `out_like` + explicit dimensions)

```idl
# Batch Normalization
kernel batch_norm(in float input, in float gamma, in float beta,
                  in float mean, in float variance, out_like input,
                  int height, int width, int channels, float epsilon);
extension res as batch_norm(resident float input, resident float gamma,
                           resident float beta, resident float mean,
                           resident float variance, resident float output,
                           int height, int width, int channels, float epsilon);
```

**Why `out_like` + dimensions?**
- Output has same shape as input → `out_like`
- But need dimensions to process per-channel parameters correctly
- gamma, beta, mean, variance are all (channels,) shaped

**Usage**:
```python
input_data = np.random.randn(64, 224, 224).astype(np.float32)

# Per-channel parameters
gamma = np.ones(64, dtype=np.float32)
beta = np.zeros(64, dtype=np.float32)
mean = np.random.randn(64).astype(np.float32)
variance = np.ones(64, dtype=np.float32)

output = prog.batch_norm(
    input_data, gamma, beta, mean, variance,
    height=224,
    width=224,
    channels=64,
    epsilon=1e-5
)
# Output shape: (64, 224, 224) - same as input
```

---

### 6. Matrix Multiplication (Explicit dimensions for 2D)

```idl
# Matrix multiplication with bias (optimized with SIMD)
# C = A * B + bias
kernel matrix_multiply_simd(in float A, in float B, in float bias, out float C,
                           int M, int N, int K);
extension res as matrix_multiply_simd(resident float A, resident float B,
                                     resident float bias, resident float C,
                                     int M, int N, int K);
```

**Why explicit M, N, K?**
- Matrix shapes: A(M×K), B(K×N), C(M×N)
- Cannot infer from buffer sizes alone
- Need to know matrix dimensions for correct multiplication

**Usage**:
```python
A = np.random.randn(128, 512).astype(np.float32)  # Batch size 128, features 512
B = np.random.randn(512, 1000).astype(np.float32)  # 512 features → 1000 classes
bias = np.zeros(1000, dtype=np.float32)

C = prog.matrix_multiply_simd(
    A, B, bias, C_output,
    M=128,   # batch size
    N=1000,  # output features
    K=512    # input features
)
# C shape: (128, 1000)
```

---

## Design Rationale Summary

| Operation | Output Shape | Parameter Hint | Reason |
|-----------|--------------|----------------|---------|
| **ReLU** | Same as input | `out_like input` | Shape preserved, idiomatic |
| **Softmax** | Same as input | `out_like input` | Shape preserved, idiomatic |
| **Fully Connected** | Different from input | `sizeof input`, `sizeof output` | Sizes differ but dimensions simple |
| **Convolution** | Complex 3D | Explicit `int` dims | Multi-dimensional tensors |
| **Pooling** | **Changes** shape | Explicit `int` dims | Output dims ≠ input dims |
| **Batch Norm** | Same + per-channel | `out_like` + explicit dims | Same shape but need channel info |
| **Matrix Multiply** | 2D with formula | Explicit `int` M, N, K | Matrix dimensions for multiplication |

---

## Comparison: Before vs After

### Before (Explicit sizes everywhere)

```idl
# ❌ Not idiomatic - explicit size for same-shape operation
kernel relu_activation(in float input, out float output, int size);
kernel softmax(in float input, out float output, int size);
```

### After (Idiomatic yapocis)

```idl
# ✅ Idiomatic - let yapocis infer size
kernel relu_activation(in float input, out_like input);
kernel softmax(in float input, out_like input);
```

**Benefits**:
- Cleaner API - no redundant size parameter
- Consistent with other yapocis interfaces (audio, imageops, etc.)
- Less error-prone - can't pass wrong size

---

## Complete Interface

```idl
interface vgg16 {
    # =================================================================
    # Convolutional Layers
    # =================================================================

    kernel conv2d_3x3(in float input, in float weights, in float bias, out float output,
                      int input_height, int input_width,
                      int input_channels, int output_channels);
    extension res as conv2d_3x3(resident float input, resident float weights,
                                resident float bias, resident float output,
                                int input_height, int input_width,
                                int input_channels, int output_channels);

    kernel conv2d_3x3_optimized(in float input, in float weights, in float bias, out float output,
                                int input_height, int input_width,
                                int input_channels, int output_channels);
    extension res as conv2d_3x3_optimized(resident float input, resident float weights,
                                          resident float bias, resident float output,
                                          int input_height, int input_width,
                                          int input_channels, int output_channels);

    # =================================================================
    # Activation Functions
    # =================================================================

    kernel relu_activation(in float input, out_like input);
    extension res as relu_activation(resident float input, resident float output);

    kernel softmax(in float input, out_like input);
    extension res as softmax(resident float input, resident float output);

    # =================================================================
    # Pooling Layers
    # =================================================================

    kernel max_pool_2x2(in float input, out float output,
                        int input_height, int input_width, int channels);
    extension res as max_pool_2x2(resident float input, resident float output,
                                   int input_height, int input_width, int channels);

    # =================================================================
    # Normalization Layers
    # =================================================================

    kernel batch_norm(in float input, in float gamma, in float beta,
                      in float mean, in float variance, out_like input,
                      int height, int width, int channels, float epsilon);
    extension res as batch_norm(resident float input, resident float gamma, resident float beta,
                                 resident float mean, resident float variance, resident float output,
                                 int height, int width, int channels, float epsilon);

    # =================================================================
    # Fully Connected (Dense) Layers
    # =================================================================

    kernel fully_connected(in float input, in float weights, in float bias, out float output,
                          sizeof input, sizeof output);
    extension res as fully_connected(resident float input, resident float weights,
                                     resident float bias, resident float output,
                                     sizeof input, sizeof output);

    kernel matrix_multiply_simd(in float A, in float B, in float bias, out float C,
                               int M, int N, int K);
    extension res as matrix_multiply_simd(resident float A, resident float B,
                                          resident float bias, resident float C,
                                          int M, int N, int K);
};
```

---

## Test Coverage

Created comprehensive functional tests in `tests/test_vgg16_kernels.py`:

### Test Classes

1. **TestActivations** (5 tests)
   - `test_relu_activation_basic` - Simple known values
   - `test_relu_activation_large` - 1000 random values
   - `test_relu_activation_3d` - 3D tensor (C×H×W)
   - `test_softmax_simple` - 3-class probabilities
   - `test_softmax_numerical_stability` - Large values

2. **TestPooling** (3 tests)
   - `test_max_pool_2x2_simple` - Known max values
   - `test_max_pool_2x2_multi_channel` - 2 channels
   - `test_max_pool_2x2_negative_values` - Negative inputs

3. **TestFullyConnected** (3 tests)
   - `test_fully_connected_simple` - 2×3 weights
   - `test_fully_connected_identity` - Identity matrix
   - `test_fully_connected_zero_bias` - No bias

4. **TestMatrixMultiply** (2 tests)
   - `test_matrix_multiply_simd_simple` - 2×2 × 2×3
   - `test_matrix_multiply_simd_identity` - With identity

5. **TestConvolution** (5 tests)
   - `test_conv2d_3x3_identity` - Identity kernel
   - `test_conv2d_3x3_uniform_filter` - Averaging
   - `test_conv2d_3x3_horizontal_edge` - Edge detection
   - `test_conv2d_3x3_multi_channel` - Multi-channel
   - `test_conv2d_3x3_vs_optimized` - Basic vs optimized

6. **TestBatchNorm** (3 tests)
   - `test_batch_norm_standardize` - Zero mean, unit var
   - `test_batch_norm_scale_shift` - Learned parameters
   - `test_batch_norm_multi_channel` - 2 channels

7. **TestIntegration** (2 tests)
   - `test_conv_relu_pool_pipeline` - Complete CNN block
   - `test_fc_relu_fc_pipeline` - Classifier pipeline

**Total: 23 functional tests**

### Running Tests

```bash
# Run all VGG16 tests
YAPOCIS_BACKEND=metal pytest tests/test_vgg16_kernels.py -v

# Run specific test class
YAPOCIS_BACKEND=metal pytest tests/test_vgg16_kernels.py::TestActivations -v

# Run single test
YAPOCIS_BACKEND=metal pytest tests/test_vgg16_kernels.py::TestActivations::test_relu_activation_basic -v
```

---

## Key Takeaways

### ✅ Use yapocis idioms where possible
- `out_like` for same-shape operations
- `sizeof` for buffer sizes
- Makes API cleaner and more consistent

### ⚠️ Use explicit dimensions when needed
- Multi-dimensional tensors (3D, 4D)
- Shape transformations (pooling, reshape)
- Matrix operations with specific dimensions

### 📐 Rule of Thumb

**Can you infer output from input alone?**
- **YES** → Use `out_like` or `sizeof`
- **NO** → Use explicit dimensions

**Examples**:
- ReLU: YES (output = input) → `out_like`
- Pool 2x2: NO (output = input/2) → explicit dims
- Fully Connected: PARTIAL (sizes differ) → `sizeof`
- Convolution: NO (3D with channels) → explicit dims

---

## Files Summary

**Created/Modified**:
1. `src/yapocis/interfaces/vgg16.interface` - Idiomatic interface definition
2. `tests/test_vgg16_kernels.py` - 23 comprehensive functional tests
3. `claude-debug/VGG16_INTERFACE_DESIGN.md` - Detailed documentation
4. `claude-debug/VGG16_FINAL_INTERFACE.md` - This summary

**Metal Kernel** (unchanged):
- `src/yapocis/kernels/metal/vgg16.mako` - Original Metal implementation

---

**Status**: ✅ Complete and idiomatic
**Date**: 2025-10-26
**Interface Version**: Final - using yapocis idioms (`out_like`, `sizeof`)
