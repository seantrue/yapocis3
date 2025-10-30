# RetinaNet Weight Conversion Summary

## Goal
Convert official torchvision RetinaNet weights to yapocis format to fix detection accuracy.

## Work Completed

### 1. Proper BatchNorm Folding
- **Problem**: Torchvision uses Conv (no bias) + BatchNorm
- **Solution**: Implemented proper BN folding formula:
  ```python
  scale = bn_weight / sqrt(bn_var + eps)
  folded_weight = conv_weight * scale
  folded_bias = bn_bias - (bn_mean * scale)
  ```
- **Applied to**: Backbone (ResNet50) stem and all 16 bottleneck blocks

### 2. Version Differences
**V1 (`retinanet_resnet50_fpn`)**:
- Backbone: Conv + BN (needs folding) ✓
- Heads: Conv + Bias (no norm) ✓

**V2 (`retinanet_resnet50_fpn_v2`)**:
- Backbone: Conv + BN (needs folding) ✓
- Heads: Conv + **GroupNorm** (can't fold!) ✗

**Decision**: Use V1 because GroupNorm can't be folded into conv weights.

### 3. Conversion Results
- Created: `src/yapocis/models/retinanet/weights_torchvision.npz`
- Size: 120.7 MB
- Tensors: 142 (backbone + FPN + heads)
- All required keys present and validated

## Test Results

### Torchvision V1 (reference)
```
Image: 000000001584.jpg (bus scene)
Top detection: bus, score=0.979, bbox=(126.7, 87.8, 525.7, 536.7)
Size: 399×449 pixels (covers entire bus)
```

### Yapocis with Converted Weights
```
Top detection: bus, score=0.880, bbox=(519.1, 339.8, 552.9, 370.3)
Size: 33×30 pixels (tiny region on bus)
IoU with torchvision: 0.001 (no overlap!)
```

### Yapocis with Original Weights
```
Top detection: bus, score=0.880, bbox=(519.1, 339.8, 552.9, 370.3)
IDENTICAL results!
```

## Conclusion

**The bug is in the yapocis implementation, not the weights.**

Evidence:
1. Both old and new (properly-folded) weights produce identical wrong results
2. Torchvision with v1 weights produces correct large bboxes
3. Yapocis with v1 weights produces wrong small bboxes

## Next Steps (Debugging yapocis Implementation)

The yapocis RetinaNet has a fundamental bug. Possible areas to investigate:

1. **Image Preprocessing** (retinanet.py:108-150)
   - Normalization: ImageNet mean/std correct?
   - Resize logic: shortest edge to 800px
   - Channel order: (C,H,W) vs (H,W,C)

2. **Anchor Generation** (retinanet.py:492-528)
   - Anchor scales: [2^0, 2^(1/3), 2^(2/3)]
   - Anchor ratios: [0.5, 1.0, 2.0]
   - Base size: 4 pixels at stride 8
   - Feature pyramid stride matching

3. **FPN Implementation** (retinanet.py:283-371)
   - Lateral connection dimensions
   - Upsampling method
   - Feature concatenation order
   - Extra levels (P6, P7) generation

4. **Detection Heads** (retinanet.py:372-489)
   - Classification head output shape
   - Regression head output shape
   - Anchor-feature correspondence

5. **Bbox Decoding** (retinanet.py:531-564)
   - Already swapped dx/dy (line 537-541)
   - Delta scaling factors
   - Coordinate clamping

6. **NMS** (retinanet.py:566-626)
   - IoU threshold
   - Score threshold
   - Per-class vs class-agnostic

## Files Created

- `convert_torchvision_weights.py` - Conversion script with BN folding
- `compare_retinanet_torchvision.py` - Side-by-side comparison
- `check_weight_keys.py` - Weight validation
- `inspect_torch_structure.py` - Check BN presence
- `inspect_v2_structure.py` - Check v2 head structure
- `RETINANET_V1_VS_V2.md` - Version differences
- `BBOX_TRANSPOSE_INVESTIGATION.md` - Earlier bbox investigation

## Recommendation

Use the Explore agent to systematically compare yapocis vs torchvision intermediate outputs:
1. Preprocessed image values
2. Backbone C3/C4/C5 feature maps
3. FPN P3/P4/P5/P6/P7 outputs
4. Classification scores (raw logits)
5. Bbox deltas (raw predictions)
6. Anchors (coordinates)
7. Decoded bboxes (after delta application)
8. NMS filtered results

This will pinpoint exactly where the implementations diverge.
