# Bug Found: FPN Divergence in yapocis RetinaNet

## Summary

The yapocis RetinaNet produces incorrect bounding boxes because **the FPN (Feature Pyramid Network) implementation has major divergence from torchvision at P3 and P4 levels.**

## Systematic Comparison Results

### 1. Preprocessing ✓
- **Status**: Perfect match
- **Max difference**: 0.000000
- Both implementations produce identical normalized images

### 2. Backbone (ResNet50) ⚠️
- **C3**: max diff 0.039 (small, acceptable)
- **C4**: max diff 0.064 (small, acceptable)
- **C5**: max diff 0.331 (moderate - likely BN folding precision)
- **Status**: Small differences, likely from BatchNorm folding numerics

### 3. FPN ✗ **BUG LOCATION**
- **P3**: max diff 0.579 ✗ **LARGE DIVERGENCE**
- **P4**: max diff 0.450 ✗ **LARGE DIVERGENCE**
- **P5**: max diff 0.036 (small)

**This is the root cause!** The small backbone differences get amplified in FPN, especially at P3 and P4.

## Impact Chain

```
Backbone (small diff) → FPN (LARGE diff at P3/P4) → Wrong predictions
```

The FPN divergence of ~0.5-0.6 at P3/P4 causes completely wrong:
- Classification scores
- Bbox delta predictions
- Final decoded bboxes (33×30 pixels instead of 399×449)

## FPN Implementation Location

File: `src/yapocis/lib/retinanet.py`
Method: `_fpn_forward()` (lines 296-371)

## Next Steps

### Investigation Areas

1. **FPN Upsampling** (lines ~320-340)
   - Check upsampling method (nearest? bilinear?)
   - Verify scale factor = 2.0
   - Check if dimensions match after upsample

2. **FPN Addition/Concatenation** (lines ~345-360)
   - Verify lateral connection dimensions
   - Check addition vs concatenation
   - Verify channel alignment

3. **FPN Convolution Order** (lines ~330-350)
   - Check if 1×1 lateral conv is applied before or after addition
   - Verify 3×3 smoothing conv placement

4. **Feature Map Ordering**
   - Torchvision: top-down (C5 → C4 → C3)
   - Yapocis: verify order matches

### Comparison with Torchvision

Torchvision FPN structure:
```python
# Inner blocks: 1×1 conv for lateral connection
backbone.fpn.inner_blocks.{0,1,2}.0.weight/bias

# Layer blocks: 3×3 conv for smoothing
backbone.fpn.layer_blocks.{0,1,2}.0.weight/bias

# Process:
# 1. P5 = lateral(C5)
# 2. P4 = lateral(C4) + upsample(P5)
# 3. P3 = lateral(C3) + upsample(P4)
# 4. Apply 3×3 smoothing to each
```

### Specific Checks

1. **Upsample P5 → P4**:
   - Input: P5 (256, 25, 25)
   - Output should be: (256, 50, 50)
   - Method: nearest or bilinear?

2. **Upsample P4 → P3**:
   - Input: P4 (256, 50, 50)
   - Output should be: (256, 100, 100)

3. **Lateral connections**:
   - C3 (512, 100, 100) → 1×1 conv → (256, 100, 100)
   - C4 (1024, 50, 50) → 1×1 conv → (256, 50, 50)
   - C5 (2048, 25, 25) → 1×1 conv → (256, 25, 25)

4. **Addition timing**:
   - Torchvision: `P4 = lateral(C4) + upsample(P5)`
   - Check yapocis matches this order

## Testing Strategy

Create a minimal test that:
1. Feeds C3, C4, C5 (from torchvision) into yapocis FPN
2. Compares yapocis FPN output with torchvision FPN output
3. Identifies which specific operation causes divergence

## Expected Fix Impact

Once FPN is fixed to match torchvision:
- P3, P4 differences should drop to <0.05
- Detection heads will get correct features
- Bboxes should match torchvision (400×450 pixels)
- IoU with torchvision should be >0.5

## Files

- Debug script: `claude-debug/debug_retinanet_internals.py`
- Output: `claude-debug/debug_output.txt`
- Implementation: `src/yapocis/lib/retinanet.py:296-371`
