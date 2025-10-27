# ResNet50 Fix Summary

## Problem
ResNet50 was producing very low confidence predictions (0.20% instead of expected >50%).

## Root Cause
**Bug in global pooling dispatch sizes** in `src/yapocis/interfaces/cnn.interface`:

### Before (WRONG):
```
kernel global_avg_pool(in float input, out float output,
                       height_of input, depth_of input, width_of input);
with { dispatch_size (width_of input); };

kernel global_max_pool(in float input, out float output,
                       height_of input, depth_of input, width_of input);
with { dispatch_size (width_of input); };
```

### After (FIXED):
```
kernel global_avg_pool(in float input, out float output,
                       height_of input, depth_of input, width_of input);
with { dispatch_size (depth_of input); };

kernel global_max_pool(in float input, out float output,
                       height_of input, depth_of input, width_of input);
with { dispatch_size (depth_of input); };
```

## Impact

For input shape (2048, 7, 7):
- **Before**: Only 7 threads dispatched (width_of input = 7)
- **After**: 2048 threads dispatched (depth_of input = 2048)

This caused the global average pooling to only compute 7 out of 2048 channels, leaving the rest as zeros. The resulting feature vector had mean=0.0023 instead of mean=0.3854 (167x smaller!).

## Related Fix: max_pool_3x3_stride2

During this investigation, we also implemented the missing `max_pool_3x3_stride2` kernel required by ResNet50:

### Files Added/Modified:
1. **Interface**: `src/yapocis/interfaces/cnn.interface` (lines 127-135)
2. **Metal**: `src/yapocis/kernels/metal/cnn.mako` (lines 295-336)
3. **OpenCL**: `src/yapocis/kernels/ocl/cnn.mako` (lines 287-326)
4. **Numbax**: `src/yapocis/kernels/numbax/cnn.py` (lines 431-503)
5. **ResNet50**: `src/yapocis/lib/resnet50.py` (line 255)

## Test Results

### Before Fix:
```
Top 5 predictions:
  1. hotdog                      0.20%
  2. carton                      0.20%
  3. projectile                  0.15%
  4. missile                     0.15%
  5. airship                     0.14%
```

### After Fix:
```
Top 5 predictions:
  1. hotdog                     64.96%
  2. pretzel                     8.91%
  3. cheeseburger                4.03%
  4. French loaf                 3.48%
  5. hot pot                     3.12%
```

### Additional Test (test.jpg - Chihuahua):
```
Top 5 predictions:
  1. Chihuahua                  53.15%
  2. Pembroke                   42.37%
  3. Pomeranian                  1.24%
  4. Cardigan                    1.18%
  5. Siberian husky              0.51%
```

## Verification

Layer-by-layer comparison with PyTorch shows exact matches:

| Layer | PyTorch | Yapocis | Status |
|-------|---------|---------|--------|
| Stem+ReLU | mean=0.2628 | mean=0.2628 | ✓ |
| MaxPool | mean=0.4070 | mean=0.4070 | ✓ |
| Layer1 | range=[0, 2.6468] | range=[0, 2.6468] | ✓ |
| Layer2 | range=[0, 3.8962] | range=[0, 3.8962] | ✓ |
| Layer3 | range=[0, 1.7754] | range=[0, 1.7754] | ✓ |
| Layer4 | range=[0, 8.2350] | range=[0, 8.2350] | ✓ |
| AvgPool | **mean=0.3854** | **mean=0.3854** | ✓ FIXED |
| FC (logits) | range=[-5.14, 12.77] | range=[-5.14, 12.77] | ✓ |

## Files Modified

1. `src/yapocis/interfaces/cnn.interface`
   - Lines 127-135: Added max_pool_3x3_stride2 interface
   - Line 143: Fixed global_avg_pool dispatch size
   - Line 152: Fixed global_max_pool dispatch size

2. `src/yapocis/kernels/metal/cnn.mako`
   - Lines 295-336: Added max_pool_3x3_stride2 kernel

3. `src/yapocis/kernels/ocl/cnn.mako`
   - Lines 287-326: Added max_pool_3x3_stride2 kernel

4. `src/yapocis/kernels/numbax/cnn.py`
   - Lines 431-503: Added max_pool_3x3_stride2 kernel

5. `src/yapocis/lib/resnet50.py`
   - Line 255: Updated to use max_pool_3x3_stride2

## Debug Scripts Created

- `claude-debug/test_maxpool3x3_stride2.py` - Validates max_pool_3x3_stride2 against PyTorch
- `claude-debug/test_conv7x7_stride2.py` - Validates conv2d_7x7_stride2 against PyTorch
- `claude-debug/test_stem_weights.py` - Validates weight fusion
- `claude-debug/test_stem_forward.py` - Validates stem forward pass
- `claude-debug/test_preprocessing.py` - Validates preprocessing
- `claude-debug/debug_resnet_full_trace.py` - Full layer-by-layer trace
- `claude-debug/debug_resnet_stem_maxpool.py` - Stem+maxpool validation

All tests pass with exact matches to PyTorch.

## Status

✅ **ResNet50 fully functional** with correct high-confidence predictions
✅ **VGG16 still working** (47.22% on hotdog)
✅ **All kernels match PyTorch** (max diff: 0.000000)
