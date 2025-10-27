# VGG16 Fully Idiomatic Interface

## Summary

Updated VGG16 interface to use **all** yapocis parameter hints: `depth_of`, `height_of`, `width_of`, `sizeof`, `out_like`.

**Status**: ✅ Fully Idiomatic
**File**: `src/yapocis/interfaces/vgg16.interface`

---

## What Changed

### Before (Explicit int parameters)

```idl
# ❌ Manual dimension passing
kernel conv2d_3x3(in float input, in float weights, in float bias, out float output,
                  int input_height, int input_width,
                  int input_channels, int output_channels);

kernel max_pool_2x2(in float input, out float output,
                    int input_height, int input_width, int channels);

kernel batch_norm(in float input, in float gamma, in float beta,
                  in float mean, in float variance, out_like input,
                  int height, int width, int channels, float epsilon);
```

### After (Idiomatic yapocis)

```idl
# ✅ Automatic dimension inference
kernel conv2d_3x3(in float input, in float weights, in float bias,
                  depth_of input, height_of input, width_of input,
                  int output_channels,
                  out float output);

kernel max_pool_2x2(in float input, depth_of input, height_of input, width_of input,
                    out float output);

kernel batch_norm(in float input, in float gamma, in float beta,
                  in float mean, in float variance,
                  depth_of input, height_of input, width_of input, float epsilon,
                  out_like input);
```

---

## Complete Parameter Hint Usage

| Kernel | Parameter Hints Used | Description |
|--------|---------------------|-------------|
| **conv2d_3x3** | `depth_of`, `height_of`, `width_of` | Input dims auto-derived from 3D tensor |
| **conv2d_3x3_optimized** | `depth_of`, `height_of`, `width_of` | Same as conv2d_3x3 |
| **relu_activation** | `out_like` | Output same shape as input |
| **softmax** | `out_like` | Output same shape as input |
| **max_pool_2x2** | `depth_of`, `height_of`, `width_of` | All dims auto-derived |
| **batch_norm** | `depth_of`, `height_of`, `width_of`, `out_like` | Dims + output shape auto-derived |
| **fully_connected** | `sizeof`, `sizeof` | 1D buffer sizes |
| **matrix_multiply_simd** | Explicit `int M, N, K` | Matrix algebra requires explicit dims |

---

## Detailed Interface Breakdown

### 1. Convolutional Layers

**Uses**: `depth_of input`, `height_of input`, `width_of input`

```idl
kernel conv2d_3x3(in float input, in float weights, in float bias,
                  depth_of input,      # input_channels (auto)
                  height_of input,     # input_height (auto)
                  width_of input,      # input_width (auto)
                  int output_channels, # Cannot infer - output differs
                  out float output);
```

**Why this works**:
- Input tensor shape: `(input_channels, input_height, input_width)`
- `depth_of input` → input_channels
- `height_of input` → input_height
- `width_of input` → input_width
- `output_channels` still explicit (cannot infer output shape from input)

**Usage**:
```python
input_img = np.random.randn(3, 224, 224).astype(np.float32)  # RGB image
weights = np.random.randn(64, 3, 3, 3).astype(np.float32)    # 64 filters
bias = np.zeros(64, dtype=np.float32)

# Yapocis automatically extracts: depth=3, height=224, width=224
output = prog.conv2d_3x3(input_img, weights, bias, output_channels=64)
```

---

### 2. Pooling Layers

**Uses**: `depth_of input`, `height_of input`, `width_of input`

```idl
kernel max_pool_2x2(in float input,
                    depth_of input,   # channels (auto)
                    height_of input,  # input_height (auto)
                    width_of input,   # input_width (auto)
                    out float output);
```

**Why this works**:
- Input tensor shape: `(channels, input_height, input_width)`
- Output shape computed internally: `(channels, input_height/2, input_width/2)`
- All dimensions derived from input
- No explicit parameters needed!

**Usage**:
```python
input_data = np.random.randn(64, 224, 224).astype(np.float32)

# Yapocis automatically extracts: depth=64, height=224, width=224
# and allocates output as (64, 112, 112)
output = prog.max_pool_2x2(input_data)
```

---

### 3. Batch Normalization

**Uses**: `depth_of input`, `height_of input`, `width_of input`, `out_like input`

```idl
kernel batch_norm(in float input, in float gamma, in float beta,
                  in float mean, in float variance,
                  depth_of input,   # channels (auto)
                  height_of input,  # height (auto)
                  width_of input,   # width (auto)
                  float epsilon,
                  out_like input);  # output same shape (auto)
```

**Why this works**:
- Input shape: `(channels, height, width)`
- Output shape: same as input → `out_like input`
- Per-channel parameters: gamma, beta, mean, variance are all `(channels,)`
- Dimensions needed to index per-channel params correctly

**Usage**:
```python
input_data = np.random.randn(64, 224, 224).astype(np.float32)

# Per-channel parameters (64 channels)
gamma = np.ones(64, dtype=np.float32)
beta = np.zeros(64, dtype=np.float32)
mean = np.random.randn(64).astype(np.float32)
variance = np.ones(64, dtype=np.float32)

# Yapocis extracts: depth=64, height=224, width=224
# and creates output with same shape
output = prog.batch_norm(input_data, gamma, beta, mean, variance, epsilon=1e-5)
```

---

### 4. Activation Functions

**Uses**: `out_like input`

```idl
kernel relu_activation(in float input, out_like input);
kernel softmax(in float input, out_like input);
```

**Why this works**:
- Element-wise operations
- Output always same shape as input
- No dimensions needed

**Usage**:
```python
input_data = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)

# No size parameter needed!
output = prog.relu_activation(input_data)
probs = prog.softmax(logits)
```

---

### 5. Fully Connected Layers

**Uses**: `sizeof input`, `sizeof output`

```idl
kernel fully_connected(in float input, in float weights, in float bias,
                      sizeof input,   # input_size (auto)
                      sizeof output,  # output_size (auto)
                      out float output);
```

**Why this works**:
- 1D buffers (flattened features)
- Input size: `sizeof input` → number of elements in input buffer
- Output size: `sizeof output` → number of elements in output buffer

**Usage**:
```python
input_features = np.random.randn(512).astype(np.float32)
weights = np.random.randn(1000, 512).astype(np.float32)  # 512 → 1000
bias = np.zeros(1000, dtype=np.float32)

# Yapocis extracts: input_size=512, output_size=1000
output = prog.fully_connected(input_features, weights, bias, output_buffer)
```

---

### 6. Matrix Multiplication (Still Explicit)

**Uses**: Explicit `int M, N, K`

```idl
kernel matrix_multiply_simd(in float A, in float B, in float bias,
                           int M, int N, int K,  # Still explicit
                           out float C);
```

**Why explicit?**
- Matrix algebra requires knowing dimensions for correct multiplication
- A(M×K) × B(K×N) = C(M×N)
- Cannot infer relationship from buffer sizes alone
- Need to know which dimension is which

**Usage**:
```python
A = np.random.randn(128, 512).astype(np.float32)  # Batch × Features
B = np.random.randn(512, 1000).astype(np.float32)  # Features × Classes
bias = np.zeros(1000, dtype=np.float32)

# Must specify matrix dimensions
C = prog.matrix_multiply_simd(A, B, bias, M=128, N=1000, K=512)
```

---

## Benefits of Idiomatic Interface

### ✅ Cleaner API

**Before**:
```python
output = prog.conv2d_3x3(
    input_img, weights, bias, output_buffer,
    input_height=224,
    input_width=224,
    input_channels=3,
    output_channels=64
)
```

**After**:
```python
output = prog.conv2d_3x3(
    input_img, weights, bias,
    output_channels=64  # Only what can't be inferred!
)
```

### ✅ Less Error-Prone

**Before**: Could pass wrong dimensions
```python
# Bug: Passed width/height swapped!
output = prog.max_pool_2x2(
    input_data, output_buffer,
    input_height=224,
    input_width=256,   # Wrong!
    channels=64
)
```

**After**: Dimensions automatically correct
```python
# Always correct - derived from actual tensor
output = prog.max_pool_2x2(input_data)
```

### ✅ Consistent with yapocis Style

Matches existing interfaces:
```idl
# From hsi.interface
kernel rgb2hsi3(depth_of rgb, in float rgb, out_like rgb);

# From graphics4.interface
kernel lines(width_of a, height_of a, inout float a, ...);

# From convolvesep.interface
kernel gaussian5(int vertical, width_of input, height_of input, in float input, out_like input);
```

---

## Comparison: What's Inferred vs What's Explicit

| Kernel | Inferred Parameters | Explicit Parameters | Reason for Explicit |
|--------|---------------------|---------------------|---------------------|
| **conv2d_3x3** | input dims (3) | output_channels | Output shape differs |
| **max_pool_2x2** | ALL dims (3) | NONE | All computable |
| **batch_norm** | ALL dims (3) | epsilon | Float constant |
| **relu_activation** | size | NONE | Same shape |
| **softmax** | size | NONE | Same shape |
| **fully_connected** | both sizes (2) | NONE | Sizes from buffers |
| **matrix_multiply** | NONE | M, N, K (3) | Matrix algebra |

---

## Complete Final Interface

```idl
interface vgg16 {
    # Convolutional Layers
    kernel conv2d_3x3(in float input, in float weights, in float bias,
                      depth_of input, height_of input, width_of input,
                      int output_channels,
                      out float output);

    kernel conv2d_3x3_optimized(in float input, in float weights, in float bias,
                                depth_of input, height_of input, width_of input,
                                int output_channels,
                                out float output);

    # Activation Functions
    kernel relu_activation(in float input, out_like input);
    kernel softmax(in float input, out_like input);

    # Pooling Layers
    kernel max_pool_2x2(in float input, depth_of input, height_of input, width_of input,
                        out float output);

    # Normalization Layers
    kernel batch_norm(in float input, in float gamma, in float beta,
                      in float mean, in float variance,
                      depth_of input, height_of input, width_of input, float epsilon,
                      out_like input);

    # Fully Connected Layers
    kernel fully_connected(in float input, in float weights, in float bias,
                          sizeof input, sizeof output,
                          out float output);

    kernel matrix_multiply_simd(in float A, in float B, in float bias,
                               int M, int N, int K,
                               out float C);

    # All kernels have resident extensions
    extension res as <kernel_name>(...);
};
```

---

## Decision Tree: Which Parameter Hint to Use?

```
Is output same shape as input?
├─ YES → Use `out_like input`
│         Examples: relu_activation, softmax, batch_norm
│
└─ NO → Does input have 3D structure (C×H×W)?
   ├─ YES → Use `depth_of`, `height_of`, `width_of`
   │         Examples: conv2d_3x3, max_pool_2x2, batch_norm
   │
   └─ NO → Is it 1D (just size)?
      ├─ YES → Use `sizeof input`, `sizeof output`
      │         Examples: fully_connected
      │
      └─ NO → Use explicit `int` parameters
                Examples: matrix_multiply_simd (M×N×K)
```

---

## Summary

✅ **Maximally idiomatic** - Uses all yapocis hints where applicable

✅ **Cleaner API** - Only explicit parameters that truly can't be inferred

✅ **Safer** - Dimensions always match actual tensors

✅ **Consistent** - Follows established yapocis patterns

The VGG16 interface now represents **best practices** for yapocis neural network interfaces!

---

**Date**: 2025-10-26
**Version**: Fully Idiomatic (using `depth_of`, `height_of`, `width_of`, `sizeof`, `out_like`)
**Status**: ✅ Complete
