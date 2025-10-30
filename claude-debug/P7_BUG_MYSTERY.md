# P7 Convolution Bug - Unsolved Mystery

## Summary

The `conv2d_3x3_stride2` kernel produces WRONG results ONLY when using:
- **REAL P7 weights** (from torchvision)
- **REAL P6 data** (from torchvision FPN)

But works PERFECTLY in all other cases.

## Test Results

| Input P6 | Weights | Result | Max Diff |
|----------|---------|--------|----------|
| Random 1ch, 5×5 | Random 1ch | ✓ WORKS | 0.000000 |
| Random 256ch, 13×13 | Random 256ch | ✓ WORKS | 0.000001 |
| Random 256ch, 13×13 | **REAL P7** | ✓ WORKS | 0.000001 |
| **REAL P6** from torch | **REAL P7** | ✗ FAILS | **2.498** |

## Observations

### Real P6 Data (from torchvision)
- Shape: (256, 13, 13)
- Dtype: float32
- Range: [-3.951, 5.984]
- Mean: 0.020, Std: 0.759
- No NaN, no Inf
- All 43,264 values non-zero
- C-contiguous with correct strides
- Only 5 values have |x| > 5

**Data looks completely normal!**

### Failure Mode
When using real P6 + real P7 weights:
- Yapocis P7: [-5.873, 7.011]
- Torch P7: [-4.607, 5.167]
- Sample (3, 3): torch=-0.453, yapocis=-0.995 (off by 2.2×)

The kernel produces values with **larger magnitude** than expected.

## Hypotheses (All Tested)

1. ✗ **Kernel bug**: Works perfectly with random data
2. ✗ **Weight loading**: Weights match exactly (diff=0.000000)
3. ✗ **Data layout**: P6 is C-contiguous with correct strides
4. ✗ **Multi-channel accumulation**: Works with 256 random channels
5. ✗ **Numerical precision**: Works with real weights + random P6
6. ✗ **Anomalous values**: No NaN/Inf, normal range
7. ❓ **GPU state/memory**: Not tested
8. ❓ **Specific value patterns**: Unknown trigger

## Impact on RetinaNet

After fixing P6/P7 size and ReLU bugs:
- P6 features close to torchvision (diff 0.054)
- P7 features still wrong (diff 2.479)
- Detection IoU = 0.004 (no improvement)
- All detections are tiny (~28×25 pixels vs 399×449)

The P7 bug prevents the model from detecting large objects.

## Files

**Test Scripts**:
- `test_stride2_minimal.py`: 1-channel test (passes)
- `test_stride2_256ch.py`: 256-channel random test (passes)
- `test_real_weights.py`: Real weights + random P6 (passes)
- `test_p7_convolution.py`: Real weights + real P6 (FAILS)

**Modified Files**:
- `src/yapocis/lib/retinanet.py`: Fixed P6/P7 sizes, removed ReLU
- `src/yapocis/kernels/ocl/cnn.mako`: Fixed output size calculation
- `src/yapocis/kernels/metal/cnn.mako`: Fixed output size calculation

## Next Steps (Suggestions)

### 1. Check GPU Buffer State
Test if the issue is related to GPU memory or buffer reuse:
```python
# Copy P6 to fresh numpy array
P6_copy = np.copy(torch_P6)
P6_fresh = np.ascontiguousarray(P6_copy)
# Test with fresh copy
```

### 2. Binary Search for Trigger Values
Find which specific P6 values cause the bug:
```python
# Replace parts of real P6 with random values
# Find smallest subset that triggers bug
```

### 3. Check Kernel Dispatch
Verify Metal/OCL kernel is dispatched with correct grid size:
```python
# Add debug prints in kernel dispatch code
# Check threadgroup dimensions
```

### 4. Test Different Backend
Try OpenCL backend instead of Metal:
```bash
YAPOCIS_BACKEND=ocl python test_p7_convolution.py
```

### 5. Compare Intermediate Values
Add debug output in kernel to check:
- Input indices being read
- Weight indices being read
- Accumulation values
- Whether all 256×3×3 values are processed

### 6. Verify Weight Memory Layout
Even though weights match, check if kernel expects different layout:
```python
# Try transposing weight dimensions
# Try different contiguity orders
```

## Conclusion

This is a **heisenbug** - it appears with real data but vanishes with synthetic data that should be equivalent. The trigger is something subtle about the specific values in the real P6 tensor, possibly interacting with:
- GPU floating-point behavior
- Memory access patterns
- Buffer management
- Compiler optimizations

The conv2d_3x3_stride2 kernel implementation appears correct (works in all synthetic tests), but there's a hidden bug triggered by real-world data patterns.
