# ROOT CAUSE FOUND: Upsampling Method Mismatch

## The Bug

**yapocis uses BILINEAR interpolation for FPN upsampling**
**torchvision uses NEAREST NEIGHBOR interpolation**

This mismatch causes large divergence in FPN P3/P4 features (0.45-0.58), leading to completely wrong bbox predictions.

## Evidence

### Torchvision FPN
File: `torchvision/ops/feature_pyramid_network.py`

```python
inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
```

### Yapocis FPN
File: `src/yapocis/kernels/metal/cnn.mako` line 825-883

```metal
// Bilinear interpolation (lines 846-879)
float in_h = (float(out_h) + 0.5f) / 2.0f - 0.5f;
float in_w = (float(out_w) + 0.5f) / 2.0f - 0.5f;
// ...bilinear math...
```

## Impact Chain

```
Backbone (small diff 0.33)
  ↓
FPN upsample (BILINEAR vs NEAREST)
  ↓
P3/P4 divergence (0.45-0.58) ← BUG HERE
  ↓
Wrong detection predictions
  ↓
Tiny bboxes (33×30 instead of 399×449)
```

## Fix Required

Change `upsample_2x` kernel to use **nearest neighbor** instead of bilinear:

### Current (WRONG - bilinear):
```metal
float in_h = (float(out_h) + 0.5f) / 2.0f - 0.5f;
float in_w = (float(out_w) + 0.5f) / 2.0f - 0.5f;
// ...bilinear interpolation...
```

### Should be (CORRECT - nearest):
```metal
uint in_h = out_h / 2;
uint in_w = out_w / 2;
uint idx = c * input_height * input_width + in_h * input_width + in_w;
output[output_idx] = input[idx];
```

## Files to Modify

1. `src/yapocis/kernels/metal/cnn.mako` - line 825-883
2. `src/yapocis/kernels/ocl/cnn.mako` - equivalent upsample_2x kernel
3. (Maybe) `src/yapocis/kernels/numba/cnn.py` - if it has upsample

## Expected Result After Fix

- P3 diff: 0.579 → <0.05
- P4 diff: 0.450 → <0.05
- Bbox predictions: 33×30 → 399×449
- IoU with torchvision: 0.001 → >0.5

## Testing

After fix, run:
```bash
python claude-debug/debug_retinanet_internals.py
python claude-debug/compare_retinanet_torchvision.py
```

Should see:
- ✓ FPN P3/P4/P5 all match (diff <0.05)
- ✓ Detection bboxes match torchvision
- ✓ IoU >0.5

## Why This Happened

Bilinear is often preferred for quality, but **torchvision specifically uses nearest for FPN** to:
1. Preserve sharp features
2. Avoid smoothing that affects detection
3. Match the original FPN paper implementation

The bug was hidden because:
- Weights were wrong initially (masked the upsample bug)
- BN folding precision differences (0.33 at C5) were blamed
- Small differences compounded through FPN

## Conclusion

Single root cause: **wrong interpolation mode in upsample_2x kernel**

Fix: Change bilinear → nearest neighbor in upsample kernels.
