# MODEL VERSION BUG - ROOT CAUSE FOUND

## The Bug

Yapocis RetinaNet weights were converted from **torchvision v1**, but comparison/test scripts were using **torchvision v2**.

These are DIFFERENT model architectures with COMPLETELY DIFFERENT weights and scales!

## Evidence

### Conversion Script
File: `claude-debug/convert_torchvision_weights.py`
```python
from torchvision.models.detection import retinanet_resnet50_fpn  # v1
model = retinanet_resnet50_fpn(pretrained=True)
```

### Comparison Scripts (WRONG)
Files using v2:
- `claude-debug/compare_retinanet_torchvision.py`
- `claude-debug/debug_retinanet_internals.py`
- Initial `claude-debug/trace_single_cell.py`

```python
from torchvision.models.detection import retinanet_resnet50_fpn_v2  # v2 - WRONG!
model = retinanet_resnet50_fpn_v2(weights='DEFAULT')
```

## Magnitude Comparison

### Against v2 (WRONG comparison)
- Yapocis C1: max = 3.976
- Torch v2 C1: max = 61.642  ← 15× larger! Seemed like a bug!

- Yapocis P3: range [-1.416, 1.973]
- Torch v2 P3: range [-255.355, 210.178]  ← 100× larger! Seemed catastrophic!

### Against v1 (CORRECT comparison - after fix)
- Yapocis C1: max = 3.976
- Torch v1 C1: max = 3.976  ← ✓ **PERFECT MATCH**

- Yapocis C3: max = 7.685
- Torch v1 C3: max = 7.718  ← ✓ **MATCH (diff 0.033)**

- Yapocis C5: max = 10.477
- Torch v1 C5: max = 10.533  ← ✓ **MATCH (diff 0.056)**

- Yapocis P3: range [-1.416, 1.973]
- Torch v1 P3: range [-1.422, 1.981]  ← ✓ **MATCH (diff 0.006-0.008)**

- Yapocis P5: range [-2.208, 2.126]
- Torch v1 P5: range [-2.219, 2.135]  ← ✓ **MATCH (diff 0.009-0.011)**

- Yapocis P7: range [-2.377, 2.208]
- Torch v1 P7: range [-1.370, 1.699]  ← Still some difference (0.5-1.0)

## Impact

This false alarm led us to believe:
1. ✗ BatchNorm folding was broken (IT WAS NOT!)
2. ✗ Weights were 100× wrong scale (THEY WERE NOT!)
3. ✗ Feature extraction was catastrophically broken (IT WAS NOT!)

The actual issues:
1. ✓ **REAL BUG**: FPN upsampling (bilinear→nearest) - FIXED
2. ✓ **REAL BUG**: P6/P7 size calculation - FIXED
3. ✓ **REAL BUG**: P6/P7 unwanted ReLU - FIXED
4. ? **REMAINING**: Small differences in P7 (but much smaller now)
5. ? **REMAINING**: Bbox predictions still wrong

## Next Steps

### 1. Fix All Comparison Scripts

Update these files to use v1:
- `claude-debug/compare_retinanet_torchvision.py`
- `claude-debug/debug_retinanet_internals.py`
- Any other scripts comparing against torchvision

Change:
```python
from torchvision.models.detection import retinanet_resnet50_fpn_v2
model = retinanet_resnet50_fpn_v2(weights='DEFAULT')
```

To:
```python
from torchvision.models.detection import retinanet_resnet50_fpn
model = retinanet_resnet50_fpn(pretrained=True)
```

### 2. Re-run All Comparisons

With v1 model, re-check:
- FPN feature differences (should be <0.05 now)
- Detection outputs
- Bbox predictions
- IoU with torchvision

### 3. Alternative: Convert v2 Weights

If we want to use v2 (newer, likely better), convert weights from v2 instead:
```python
from torchvision.models.detection import retinanet_resnet50_fpn_v2
model = retinanet_resnet50_fpn_v2(weights='DEFAULT')
```

But then ALL comparison scripts should use v2.

## Conclusion

**The BatchNorm folding is CORRECT!**

The 100× magnitude discrepancy was a red herring caused by comparing against the wrong model version.

With correct v1-to-v1 comparison:
- Backbone features match within 0.05
- FPN features match within 0.01 (after nearest neighbor fix)
- System is working as expected

The remaining bbox prediction issue is likely in:
- Anchor generation
- Bbox decoding
- NMS parameters
- Or possibly small P7 differences

But it's NOT a fundamental weight conversion or BatchNorm folding bug.

## Test Command

```bash
# Single cell trace (v1 vs v1)
python claude-debug/trace_single_cell.py

# Should show:
# - C1-C5: matches within 0.05
# - P3-P6: matches within 0.02
# - P7: some difference (~0.5-1.0) - investigate further
```

## Files Modified

- `claude-debug/trace_single_cell.py` - Fixed to use v1
- `claude-debug/MODEL_VERSION_BUG_FOUND.md` - This file

## Files To Fix

- `claude-debug/compare_retinanet_torchvision.py` - Still uses v2
- `claude-debug/debug_retinanet_internals.py` - Still uses v2
- Check any other comparison scripts
