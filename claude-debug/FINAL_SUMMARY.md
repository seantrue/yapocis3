# RetinaNet Debugging Final Summary

## Accomplishments

### 1. Weight Conversion ✓
- Created proper BatchNorm folding for torchvision → yapocis conversion
- Handles backbone (ResNet50) with 16 bottleneck blocks
- Properly folds BN parameters: `W_new = W * (γ / √(σ² + ε))`, `b_new = β - μ * scale`
- Total: 142 tensors, 120.7 MB
- File: `src/yapocis/models/retinanet/weights_torchvision.npz`

### 2. Found FPN Bug ✓
- **Root cause**: Bilinear interpolation vs Nearest neighbor
- Torchvision uses `mode="nearest"` for FPN upsampling
- Yapocis used bilinear interpolation
- **Impact**: P3/P4 divergence of 0.45-0.58

### 3. Fixed FPN Bug ✓
- Modified `upsample_2x` kernel in Metal and OCL
- Changed from bilinear to nearest neighbor
- **Result**: P3/P4 differences dropped 18× (0.45 → 0.025)
- Files modified:
  - `src/yapocis/kernels/metal/cnn.mako` line 825-855
  - `src/yapocis/kernels/ocl/cnn.mako` line 798-827

## Remaining Issue

### Small Bbox Problem
After FPN fix, yapocis still produces small bboxes:
- **Yapocis**: All detections 30-60 pixels (e.g., 519×339 to 553×370)
- **Torchvision**: Large bus 400×450 pixels (127×88 to 526×537)
- **IoU**: Still 0.001 (no overlap)

### Observations
1. FPN features now match (diff <0.03) ✓
2. Classification scores similar (0.917 vs 0.979) ✓
3. But bboxes completely different ✗
4. Yapocis only detects in region x:190-570, y:300-400
5. Missing the large full-bus detection

### Likely Causes
1. **Anchor generation** - wrong sizes/aspect ratios?
2. **Pyramid level usage** - P5/P6/P7 not contributing?
3. **Bbox decoding** - beyond dx/dy swap, something else wrong?
4. **NMS** - suppressing large detections?

## Next Steps

### Debug Anchors
1. Check anchor base sizes at each pyramid level
2. Verify anchor scales: [2^0, 2^(1/3), 2^(2/3)]
3. Verify anchor ratios: [0.5, 1.0, 2.0]
4. Check if P5/P6/P7 anchors are generated correctly

### Debug Bbox Decoding
1. Compare raw bbox deltas (before decoding)
2. Check delta scaling factors
3. Verify anchor center calculation
4. Check if coordinates are clamped correctly

### Debug Detection Matching
1. See which pyramid levels produce the top detections
2. Check if large anchors (P5/P6/P7) are scoring high
3. Verify NMS isn't suppressing large bboxes

## Files Created

**Documentation**:
- `RETINANET_WEIGHT_CONVERSION_SUMMARY.md` - Weight conversion process
- `RETINANET_V1_VS_V2.md` - Version differences
- `BUG_FOUND_FPN_DIVERGENCE.md` - FPN bug discovery
- `BUG_ROOT_CAUSE_FOUND.md` - Upsampling bug root cause
- `FINAL_SUMMARY.md` - This file

**Scripts**:
- `convert_torchvision_weights.py` - Weight converter with BN folding
- `compare_retinanet_torchvision.py` - Side-by-side comparison
- `debug_retinanet_internals.py` - Internal state comparison
- `check_weight_keys.py` - Weight validation
- `inspect_torch_structure.py` - Check BN structure
- `inspect_v2_structure.py` - Check v2 heads

**Test Data**:
- Downloaded 10 COCO validation images
- `coco_test_images/000000001584.jpg` - Main test image (bus scene)

## Diagnosis Timeline

1. ✓ Weights were wrong → Converted with BN folding
2. ✓ Still wrong → Found bilinear vs nearest bug
3. ✓ Fixed upsampling → FPN now matches
4. ✗ Still wrong bboxes → New bug in anchors/decoding

## Current Status

**What works**:
- ✓ Image preprocessing
- ✓ Backbone (ResNet50) - small diffs from BN folding
- ✓ FPN (after nearest neighbor fix)
- ✓ Classification scores reasonable

**What doesn't work**:
- ✗ Bbox predictions - all too small
- ✗ Missing large anchor detections
- ✗ IoU with torchvision = 0.001

## Test Command

```bash
# Compare implementations
python claude-debug/compare_retinanet_torchvision.py

# Debug internals
python claude-debug/debug_retinanet_internals.py

# Run detection
python examples/retinanet.py \
  --weights src/yapocis/models/retinanet/weights_torchvision.npz \
  claude-debug/coco_test_images/000000001584.jpg
```

## Conclusion

Major progress:
1. Proper weight conversion with BN folding
2. Fixed FPN upsampling (bilinear → nearest)
3. FPN features now match torchvision

But bbox predictions still wrong - likely issue with:
- Anchor generation/scaling
- Bbox decoding beyond dx/dy swap
- Pyramid level contribution to final detections

The bug has been isolated to the detection/anchor system, not the feature extraction.
