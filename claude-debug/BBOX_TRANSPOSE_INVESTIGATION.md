# RetinaNet Bounding Box Transpose Investigation

## Issue
User reported that bounding boxes appear to be transposed (x-y coordinates swapped).

## Investigation Results

### 1. CNN Output Shape ✓
Tested with non-square image (400x600 → 800x1200):
- **Result**: CNN correctly outputs (C, H, W) format
- Input: 800x1200 → C3: (512, 100, 150) ✓
- No transpose issue at CNN level

### 2. Bbox Delta Order
The network outputs 4 deltas per anchor: (delta0, delta1, delta2, delta3)

**Original interpretation** (retinanet.py:537-540):
```python
dx = bbox_deltas[:, 0]
dy = bbox_deltas[:, 1]
dw = bbox_deltas[:, 2]
dh = bbox_deltas[:, 3]
```

**Swapped interpretation** (current fix):
```python
dy = bbox_deltas[:, 0]  # Swapped!
dx = bbox_deltas[:, 1]  # Swapped!
dh = bbox_deltas[:, 2]  # Swapped!
dw = bbox_deltas[:, 3]  # Swapped!
```

### 3. Test Results

#### Zebra Image (000000001584.jpg, 612x612)

**Before swap:**
- Top detection: (523.4, 335.5, 553.9, 369.3)
- Size: 30.6 × 33.8 pixels

**After swap:**
- Top detection: (519.1, 339.8, 552.9, 370.3)
- Size: 33.8 × 30.6 pixels

### 4. Visualization Files Generated

Two versions created for comparison:
1. **`claude-debug/bbox_normal.jpg`** - Current interpretation (with swap fix)
2. **`claude-debug/bbox_transposed.jpg`** - Alternative (coordinates swapped in drawing)

## Current Status

✓ CNN output shape is correct (C, H, W)
✓ Anchor generation order matches prediction order
? Bbox delta order - need visual confirmation

## Next Steps

**Please review the visualization files:**
- `claude-debug/bbox_normal.jpg` (RED boxes)
- `claude-debug/bbox_transposed.jpg` (BLUE boxes)

Which set of bounding boxes correctly aligns with objects in the image?

## Notes

- Still detecting "bus" instead of "zebra" - this suggests weights may not be properly trained
  or there's a class mapping issue
- COCO labels file had typo ("cofa" → "couch") which was fixed
- Downloaded 10 COCO validation images to `claude-debug/coco_test_images/`

## Files Modified

- `src/yapocis/lib/retinanet.py` - Swapped bbox delta interpretation (lines 537-541)
- `src/yapocis/models/retinanet/coco_classes.txt` - Fixed "cofa" → "couch"
- `examples/retinanet.py` - Added `--test` and `--max-images` flags, queue output

## Test Files Created

- `claude-debug/download_coco_test_images.py` - Downloads COCO validation images
- `claude-debug/test_retinanet_bbox_order.py` - Visualizes both interpretations
- `claude-debug/test_retinanet_shapes.py` - Checks CNN output dimensions
- `claude-debug/test_nonsquare.jpg` - Non-square test image (400x600)
