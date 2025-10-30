# P6/P7 Fix Summary

## Bugs Found and Fixed

### 1. Wrong Output Size Calculation ✓ FIXED

**Problem**: P6/P7 were using integer division without accounting for padding:
```python
# WRONG
P6 = np.zeros((256, H5 // 2, W5 // 2))  # 25 // 2 = 12
```

**Fix**: Use correct formula for padding=1, kernel=3, stride=2:
```python
# CORRECT
H6 = (H5 - 1) // 2 + 1  # (25 - 1) // 2 + 1 = 13
P6 = np.zeros((256, H6, W6))
```

**Result**:
- P6: (256, 12, 12) → (256, 13, 13) ✓
- P7: (256, 6, 6) → (256, 7, 7) ✓

### 2. Unwanted ReLU Activation ✓ FIXED

**Problem**: yapocis applied ReLU after P6/P7, but torchvision doesn't:
```python
# WRONG
P6 = self.prog.conv2d_3x3_stride2(P5, p6_W, p6_b, P6, 256)
P6 = self.prog.relu_activation(P6)  # ← Unwanted!
```

**Fix**: Remove ReLU (torchvision's LastLevelP6P7 has no activation):
```python
# CORRECT
P6 = self.prog.conv2d_3x3_stride2(P5, p6_W, p6_b, P6, 256)
# No ReLU!
```

**Result**:
- Yapocis P6: [0.000, 5.989] → [-3.945, 5.989] ✓ (now has negatives)
- Matches torchvision P6: [-3.951, 5.984] ✓

### 3. Kernel Bounds Check ✓ FIXED

**Problem**: OCL/Metal kernels calculated wrong output size:
```c
// WRONG
const uint output_height = input_height / 2;
```

**Fix**: Use correct formula:
```c
// CORRECT
const uint output_height = (input_height - 1) / 2 + 1;
```

**Files Modified**:
- `src/yapocis/lib/retinanet.py`: Lines 381-399
- `src/yapocis/kernels/ocl/cnn.mako`: Lines 706-708
- `src/yapocis/kernels/metal/cnn.mako`: Lines 732-734

## Remaining Problem: P7 Features Still Wrong

### Current Status

After fixes, feature comparison:
- ✓ P3: max_diff=0.032 (MATCH)
- ✓ P4: max_diff=0.025 (MATCH)
- ✓ P5: max_diff=0.036 (MATCH)
- ⚠ P6: max_diff=0.054 (close but not perfect)
- ✗ **P7: max_diff=2.479 (WRONG!)**

P7 ranges:
- Yapocis: [-5.865, 7.014]
- Torchvision: [-4.607, 5.167]

Yapocis P7 has larger absolute values, suggesting:
1. Accumulation error from P6 → P7
2. Possible kernel implementation issue
3. Possible weight mapping issue for P7

### Detection Impact

Even with P6/P7 shape fixes:
- IoU = 0.004 (no improvement)
- Yapocis detections: ~28×25 pixels
- Torchvision detections: 399×449 pixels (full bus)

The P7 feature errors are cascading through the detection heads, producing:
- Very high scores on P7 (0.999) but at wrong locations
- Wrong bbox predictions

## Next Steps

### Investigate P7 Convolution

1. **Check P6→P7 weight values**
   - Are fpn_extra_blocks_p7_weight/bias loaded correctly?
   - Compare with torchvision's p7.weight

2. **Debug conv2d_3x3_stride2 kernel**
   - Test with simple known input/output
   - Check if padding is applied correctly for P7 (13×13 → 7×7)
   - Verify stride and offset calculations

3. **Compare intermediate P7 values**
   - Run yapocis and torchvision P6→P7 step
   - Check specific locations (e.g., (0, 2, 2))
   - Find first divergence point

### Alternative: Check Head Weights

If P7 features can't be fixed easily, check if the issue is in detection heads:
- Classification head weights for P6/P7
- Regression head weights for P6/P7
- Anchor generation for P6/P7 levels

## Test Commands

```bash
# Compare P6/P7 features
python claude-debug/compare_p6_p7.py

# Test detection
python claude-debug/debug_anchors_and_decoding.py

# Full comparison
python claude-debug/compare_retinanet_torchvision.py
```
