# HSI NaN Bug Fix Summary

## Problem
OCL and Metal versions of `hsi2rgb3` and `rgb2hsi3` produced NaN values in the alpha channel when processing 4-channel images.

## Root Cause
The kernels used `depth_of` to determine the channel stride, which correctly handled 3 or 4 channel images. However, the kernels only wrote to the first 3 channels (RGB/HSI), leaving the 4th channel (alpha) uninitialized in the output buffer.

Since `out_like` uses `np.empty_like()` to allocate output buffers (for performance), uninitialized memory could contain garbage values, including NaN.

## Why Not Use `zeros_like()`?
Using `np.zeros_like()` would be slower as it initializes all memory to zero. Since the kernels process millions of pixels, this overhead would be significant. The correct solution is to make the kernels handle all channels properly.

## Solution
Modified all kernel implementations to copy the alpha channel through when `depth > 3`:

### Files Modified:
1. **yapocis/kernels/hsi.mako** (OpenCL)
   - `rgb2hsi3`: Added alpha passthrough
   - `hsi2rgb3`: Added alpha passthrough

2. **yapocis/metalkernels/hsi.mako** (Metal)
   - `rgb2hsi3`: Added alpha passthrough
   - `hsi2rgb3`: Added alpha passthrough

3. **yapocis/cukernels/hsi.mako** (CUDA)
   - `rgb2hsi3`: Added alpha passthrough
   - `hsi2rgb3`: Added alpha passthrough

4. **yapocis/numbax/kernels/hsi.py** (Numba/NumPy)
   - `_rgb2hsi_glsl()`: Added alpha copy
   - `_hsi2rgb_glsl()`: Added alpha copy

### Code Pattern:
```c
// OpenCL/Metal/CUDA
output[i] = c.x;
output[i+1] = c.y;
output[i+2] = c.z;
if (depth > 3) {
    output[i+3] = input[i+3];  // Copy alpha channel through
}
```

```python
# Python/NumPy
out[..., :3] = rgb_values
if input.shape[-1] > 3:
    out[..., 3:] = input[..., 3:]  # Copy alpha and beyond
```

## Verification
Tested with all available backends:
- ✓ OCL: No NaN values
- ✓ Metal: No NaN values
- ✓ Numbax: No NaN values

The fix ensures that:
1. 3-channel images work as before (no extra processing)
2. 4-channel images properly preserve the alpha channel
3. No performance penalty for the common 3-channel case
4. Minimal overhead for 4-channel case (single conditional + copy)

## Test Case
The issue was discovered in `tests/test_colormath.py::test_hsi` when processing a 4-channel RGBA image. The test now passes on all backends with no NaN values.
