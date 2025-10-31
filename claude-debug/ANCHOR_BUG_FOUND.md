# ANCHOR GENERATION BUG FOUND

## Critical Discovery

Yapocis and torchvision use **DIFFERENT anchor generation methods**!

## Yapocis Anchor Generation

Uses **continuous scales** with formula: `base_size * scale * sqrt(ratio)`

```python
anchor_scales = [1.0, 1.26, 1.59]  # 2^0, 2^(1/3), 2^(2/3)
anchor_ratios = [0.5, 1.0, 2.0]

# For P3 (base=32):
sizes = [32*1.0, 32*1.26, 32*1.59] = [32.0, 40.3, 50.8]
```

**Generated P3 anchors:** 22.6-71.8 pixels
**Generated P7 anchors:** 362.0-1149.4 pixels

## Torchvision Anchor Generation

Uses **DISCRETE sizes** directly specified:

```python
Sizes: ((32, 40, 50), (64, 80, 101), (128, 161, 203), (256, 322, 406), (512, 645, 812))
Aspect ratios: ((0.5, 1.0, 2.0), ...)

# For P3:
sizes = [32, 40, 50]  # Exact integers, NOT 32*scales
```

**Actual torchvision sizes:** `[32, 40, 50, 64, 80, 101, 128, 161, 203, 256, 322, 406, 512, 645, 812]`

## The Difference

| Level | Yapocis Sizes | Torchvision Sizes | Match? |
|-------|---------------|-------------------|--------|
| P3 | [32.0, 40.3, 50.8] | [32, 40, 50] | ≈ Close |
| P4 | [64.0, 80.6, 101.6] | [64, 80, 101] | ≈ Close |
| P5 | [128.0, 161.3, 203.2] | [128, 161, 203] | ≈ Close |
| P6 | [256.0, 322.5, 406.4] | [256, 322, 406] | ≈ Close |
| P7 | [512.0, 645.1, 812.7] | [512, 645, 812] | ≈ Close |

The sizes are APPROXIMATELY the same, but torchvision uses integer rounding!

## But This Isn't The Bug

The anchor sizes are nearly identical. The small differences (e.g., 40.3 vs 40) shouldn't cause 100× bbox errors.

## Real Findings from Analysis

###  ✓ All Pyramid Levels Working

Classification scores show ALL levels producing detections:
- P3: max score 0.676
- P4: max score 0.714
- P5: max score 0.565
- P6: max score 0.715
- **P7: max score 0.731** ← Highest!

P7 is NOT broken - it's producing the strongest signal!

### ✓ Regression Deltas Look Reasonable

All delta ranges are reasonable (~-0.9 to +0.9):
- dx/dy: ~±0.4 (position adjustments)
- dw/dh: ~±1.0 (size adjustments in log space)

### The Mystery

If:
- Anchors are generated correctly ✓
- All pyramid levels work ✓
- P7 has large anchors (362-1149 pixels) ✓
- P7 has strongest classification score (0.731) ✓
- Regression deltas are reasonable ✓

**Then why are all yapocis detections tiny (28-44 pixels)?**

## Hypothesis: Bbox Decoding Bug

The issue must be in `_decode_boxes()` - converting deltas to final boxes.

### Standard Bbox Decoding Formula

```python
# Anchor: (x1, y1, x2, y2)
anchor_w = x2 - x1
anchor_h = y2 - y1
anchor_cx = (x1 + x2) / 2
anchor_cy = (y1 + y2) / 2

# Deltas: (dx, dy, dw, dh)
# Apply deltas:
pred_cx = anchor_cx + dx * anchor_w
pred_cy = anchor_cy + dy * anchor_h
pred_w = anchor_w * exp(dw)
pred_h = anchor_h * exp(dh)

# Convert back to corners:
pred_x1 = pred_cx - pred_w / 2
pred_y1 = pred_cy - pred_h / 2
pred_x2 = pred_cx + pred_w / 2
pred_y2 = pred_cy + pred_h / 2
```

### Possible Bugs

1. **Missing exponential**: `pred_w = anchor_w * dw` instead of `anchor_w * exp(dw)`
   - Result: dw=-0.5 gives width *= -0.5 (NEGATIVE!)
   - Should be: width *= exp(-0.5) = 0.606

2. **Wrong scaling**: `dx` not multiplied by `anchor_w`
   - Result: dx=0.3 moves 0.3 pixels instead of 0.3 * anchor_width

3. **Delta order wrong**: Swapped dx/dy or dw/dh

4. **Missing scaling factors**: Torchvision may use `weights=[10, 10, 5, 5]` to scale deltas

## Next Step

Check yapocis `_decode_boxes()` implementation and compare with torchvision.

## Files

- Analysis script: `claude-debug/debug_anchors_detailed.py`
- Output: `claude-debug/anchor_analysis_output.txt`
- Implementation: `src/yapocis/lib/retinanet.py:520`
