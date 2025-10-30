# Single Cell Trace Findings - CRITICAL BUG FOUND

## Test Setup

Created an 800×800 image with a single pixel set to 1.0 at position (400, 400).
Traced through yapocis and torchvision RetinaNet pipelines block by block.

## CRITICAL FINDING: Magnitude Discrepancy in Backbone

### Backbone Output Magnitudes

**Yapocis (with folded BatchNorm):**
- C1 (stem): max = 3.976, range [0.000, 3.976]
- C2 (layer1): NOT TESTED (full forward used)
- C3 (layer2): max = 7.685, range [0.000, 7.685]
- C4 (layer3): max = 5.334, range [0.000, 5.334]
- C5 (layer4): max = 10.477, range [0.000, 10.477]

**Torchvision (with BatchNorm layers):**
- C1 (stem): max = 61.642, range [0.000, 61.642]  ← **15× LARGER**
- C2 (layer1): max = 1406.306, range [0.000, 1406.306]  ← **HUGE**
- C3 (layer2): max = 1651.106, range [0.000, 1651.106]  ← **215× LARGER than yapocis**
- C4 (layer3): max = 291.879, range [0.000, 291.879]  ← **55× LARGER than yapocis**
- C5 (layer4): max = 215.440, range [0.000, 215.440]  ← **21× LARGER than yapocis**

### FPN Output Magnitudes

**Yapocis:**
- P3: range [-1.416, 1.973]
- P4: range [-1.354, 1.438]
- P5: range [-2.208, 2.126]
- P6: range [-2.144, 3.399]
- P7: range [-2.377, 2.208]

**Torchvision:**
- P3: range [-255.355, 210.178]  ← **100× LARGER**
- P4: range [-88.438, 103.345]  ← **60× LARGER**
- P5: range [-33.572, 41.525]  ← **15× LARGER**
- P6: range [-79.955, 69.538]  ← **20× LARGER**
- P7: range [-69.225, 59.317]  ← **25× LARGER**

## Root Cause: MODEL VERSION MISMATCH!

**THE BUG: Conversion script uses v1, but trace uses v2!**

- `convert_torchvision_weights.py` loads: `retinanet_resnet50_fpn` (v1)
- `trace_single_cell.py` loads: `retinanet_resnet50_fpn_v2` (v2)

These are DIFFERENT model architectures with DIFFERENT weights!

The **BatchNorm folding** is probably correct, but we're comparing against the WRONG MODEL.

### Symptoms
1. All yapocis backbone outputs are 10-200× smaller than torchvision
2. This scales through to FPN (20-100× smaller)
3. This completely breaks detection (anchors decoded with wrong scale)

### Why Detections Are Wrong

When bbox deltas are decoded with features that are 100× too small:
- Anchor adjustments become tiny
- Predicted boxes stay near anchor sizes
- Large objects cannot be detected (anchors too small relative to features)
- Only tiny detections appear

### BatchNorm Folding Formula

Current implementation uses:
```python
W_new = W * (γ / √(σ² + ε))
b_new = β - μ * (γ / √(σ² + ε))
```

But the magnitude difference suggests either:
1. The folding formula is WRONG
2. The folded weights are not being applied correctly
3. There's a scaling factor missing

### Expected Behavior

With correct BatchNorm folding:
- Folded weights should produce THE SAME output as torchvision's BatchNorm
- Output magnitudes should match exactly
- Current difference (10-200×) is catastrophic

## Impact

This bug explains:
- ✗ Tiny bbox predictions (33×30 instead of 399×449)
- ✗ Missing large objects
- ✗ Wrong detection scores
- ✗ Low IoU with torchvision (0.001-0.004)

The FPN bug (bilinear→nearest) was real but SECONDARY.
The PRIMARY bug is BatchNorm folding.

## Next Steps

### 1. Verify BatchNorm Folding Math

Check the conversion script `convert_torchvision_weights.py`:
- Verify folding formula
- Check if γ (gamma) and β (beta) are extracted correctly
- Verify running_mean (μ) and running_var (σ²) are correct
- Check epsilon value matches

### 2. Test Folding Directly

Create a minimal test:
```python
# Input: known tensor
# Run through torchvision Conv+BN
# Run through yapocis with folded weights
# Compare outputs (should be identical)
```

### 3. Alternative: Don't Fold BatchNorm

Instead of folding, keep BN parameters separate and apply them:
```python
# After conv
x = conv2d(input, W, b)
# Apply BN
x = (x - μ) / √(σ² + ε) * γ + β
```

This would require adding a BatchNorm kernel to yapocis.

## Conclusion

**THE ROOT CAUSE OF RETINANET FAILURE: INCORRECT BATCHNORM FOLDING**

The weight conversion is producing weights that are 10-200× too small in magnitude.
This cascades through the entire network, making all features too small.
When bbox deltas are decoded, the scale is completely wrong.

Fix the BatchNorm folding, and detection should work.

## Files

- Test script: `claude-debug/trace_single_cell.py`
- Output: `claude-debug/trace_single_cell_output.txt`
- Conversion script: `claude-debug/convert_torchvision_weights.py`
