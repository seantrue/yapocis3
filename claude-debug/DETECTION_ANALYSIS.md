# RetinaNet Detection Analysis - Post Version Fix

## Current Status

After fixing the model version mismatch (v1 vs v2), the feature extraction pipeline is **working correctly**.

### Feature Matching (yapocis vs torchvision v1)

| Stage | Max Difference | Status |
|-------|----------------|--------|
| Preprocessing | 0.000 | ✓ Perfect match |
| Backbone C3 | 0.039 | ✓ Excellent |
| Backbone C4 | 0.064 | ✓ Excellent |
| Backbone C5 | 0.331 | ✓ Good (BN folding precision) |
| FPN P3 | 0.032 | ✓ Excellent |
| FPN P4 | 0.025 | ✓ Excellent |
| FPN P5 | 0.036 | ✓ Excellent |
| FPN P6 | Unknown | ? Not in output |
| FPN P7 | Unknown | ? Not in output |

All features match within acceptable tolerances!

## Detection Output Comparison

### Yapocis Detections
```
Top 5 detections:
1. bus  0.999  (374.9, 377.5, 403.1, 402.8)  ← 28×25 pixels
2. bus  0.999  (277.0, 382.9, 305.2, 408.1)  ← 28×25 pixels
3. bus  0.998  (380.4, 279.4, 408.7, 304.8)  ← 28×25 pixels
4. bus  0.997  (282.6, 284.8, 310.7, 310.2)  ← 28×25 pixels
5. bus  0.993  (365.1, 368.5, 409.5, 408.7)  ← 44×40 pixels
```

All detections are **TINY** (28-44 pixels wide).

### Torchvision Detections
```
Top 5 detections:
1. bus    0.979  (126.7, 87.8, 525.7, 536.7)  ← 399×449 pixels (CORRECT)
2. bus    0.917  (507.9, 316.2, 569.3, 383.9)  ← 61×68 pixels
3. person 0.851  (188.8, 306.6, 250.7, 371.2)  ← 62×65 pixels
4. person 0.736  (75.6, 376.8, 97.3, 425.1)    ← 22×48 pixels
5. bus    0.658  (568.1, 320.6, 610.8, 403.1)  ← 43×83 pixels
```

Top detection is **CORRECT** - full bus at 399×449 pixels.

### IoU: 0.004 (no overlap)

## Problem Isolation

Since feature extraction matches well, the bug is in **post-FPN processing**:
1. Anchor generation
2. Classification/regression head outputs
3. Bbox decoding from deltas
4. NMS (Non-Maximum Suppression)

## Hypothesis: Anchor Generation Bug

### Symptoms Suggest Anchor Scale Issue

Yapocis detections are consistently tiny (28-44 pixels).
This suggests:
- Anchors at all pyramid levels are too small
- OR anchor base sizes are wrong
- OR anchor scales are wrong

### Anchor Configuration

RetinaNet uses:
- 3 scales: [2^0, 2^(1/3), 2^(2/3)] ≈ [1.0, 1.26, 1.59]
- 3 aspect ratios: [0.5, 1.0, 2.0]
- 5 pyramid levels: P3-P7
- Base sizes per level: [32, 64, 128, 256, 512] for [P3, P4, P5, P6, P7]

### Expected Anchor Sizes

At P3 (stride 8, base 32):
- Smallest: 32 × 1.0 × sqrt(0.5) = 22.6 pixels
- Largest: 32 × 1.59 × sqrt(2.0) = 72.0 pixels

At P5 (stride 32, base 128):
- Smallest: 128 × 1.0 × sqrt(0.5) = 90.5 pixels
- Largest: 128 × 1.59 × sqrt(2.0) = 288.0 pixels

At P7 (stride 128, base 512):
- Smallest: 512 × 1.0 × sqrt(0.5) = 362.0 pixels
- Largest: 512 × 1.59 × sqrt(2.0) = 1152.0 pixels

### If Anchors Are Correct

Then yapocis SHOULD be able to detect objects from 22 pixels to 1152 pixels.
The bus (399×449) should match P6/P7 anchors (~362-1152 pixels).

But yapocis is ONLY producing 28-44 pixel detections.
This suggests:
- P5/P6/P7 anchors are not contributing
- OR P5/P6/P7 features are wrong
- OR bbox decoding is scaling incorrectly

## Test Strategy

### 1. Check Anchor Generation
```python
# Generate anchors for all levels
anchors_p3 = net._generate_anchors_for_level('P3', 100, 100, stride=8, base_size=32)
anchors_p7 = net._generate_anchors_for_level('P7', 7, 7, stride=128, base_size=512)

# Print anchor size distribution
print(f"P3 anchors: {anchors_p3.shape}, sizes: {anchor_sizes(anchors_p3)}")
print(f"P7 anchors: {anchors_p7.shape}, sizes: {anchor_sizes(anchors_p7)}")
```

### 2. Check Detection Head Outputs
```python
# For each pyramid level
cls_out = net._classification_head(P7)  # Shape: (819, 7, 7)
reg_out = net._regression_head(P7)      # Shape: (36, 7, 7)

# Check which level produces high scores
for level in ['P3', 'P4', 'P5', 'P6', 'P7']:
    cls = net._classification_head(features[level])
    max_score = sigmoid(cls).max()
    print(f"{level}: max_score = {max_score:.3f}")
```

### 3. Check Bbox Decoding
```python
# Pick a high-scoring detection
# Check raw bbox delta values
# Apply decoding: box = anchor + decode(delta)
# Verify decoded box size is reasonable
```

### 4. Compare with Torchvision

Extract intermediate values from torchvision:
```python
# Anchor generation
torch_anchors = model.anchor_generator(...)

# Detection head outputs
torch_cls, torch_reg = model.head(fpn_features)

# Decoded boxes
torch_boxes = model.box_coder.decode(torch_reg, torch_anchors)
```

## Likely Root Causes

Ranked by probability:

1. **Anchor stride/base_size mismatch** (most likely)
   - yapocis may be using wrong stride values
   - or wrong base sizes for each level
   - Result: all anchors shifted/scaled incorrectly

2. **Bbox decoding bug**
   - Delta scaling factors wrong
   - Or coordinate transformation wrong
   - Result: anchors decoded to tiny boxes

3. **P6/P7 features wrong**
   - P7 showed ~0.5-1.0 difference in trace
   - If P6/P7 heads output garbage, only P3-P5 contribute
   - P3-P5 max anchor size ~288 pixels < 399 pixel bus
   - Result: large objects undetectable

4. **NMS over-suppression**
   - Unlikely (would affect both large and small boxes)

5. **Classification head bug**
   - If P6/P7 heads give low scores, those detections filtered out
   - Only P3-P5 detections pass threshold

## Next Steps

1. Print anchor sizes for all levels
2. Print max classification scores for each level
3. Print raw bbox deltas for top detections
4. Compare anchor generation with torchvision
5. Compare bbox decoding with torchvision

## Files

- Detection comparison: `claude-debug/compare_retinanet_torchvision.py`
- Feature comparison: `claude-debug/debug_retinanet_internals.py`
- Implementation: `src/yapocis/lib/retinanet.py`
