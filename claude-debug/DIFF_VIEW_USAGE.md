# Difference Visualization - Quick Start Guide

## What is it?

A 3-panel view in imagewatcher that shows:
- **Left**: Current test output
- **Middle**: Validation reference
- **Right**: Color-coded difference heat map

## How to use it

### Step 1: Generate test data (already done!)
```bash
# Creates 5 test images with validation references
python claude-debug/demo_diff_view.py        # Graphics demo
python claude-debug/test_diff_view.py        # Comprehensive tests
```

### Step 2: Launch imagewatcher
```bash
yapocis-imagewatcher
```

### Step 3: View differences
- Press **F8** to toggle difference view
- Use **a/d** to navigate between test images
- Press **F9** to toggle validation-only view
- Press **h** for help

## Test Images Available

All test images are in `queue/rendered/` with validation in `queue/validation/`:

1. **graphics_test** - Realistic graphics with subtle differences
   - Shifted circle (3px)
   - Color change
   - Line thickness difference
   - Max diff: 225.0

2. **gradient_test** - Grayscale with noise
   - Gaussian noise on gradient
   - Max diff: 20.0

3. **color_squares** - RGB color changes
   - Localized color modifications
   - Max diff: 255.0

4. **float_noise** - Very small differences
   - Tests float32 handling
   - Max diff: 0.004

5. **alpha_gradient** - RGBA with alpha differences
   - Tests 4-channel images
   - Max diff: 0.109

## Color Scale

The difference heat map uses:
- 🔵 **Blue** = No difference (0%)
- 🔷 **Cyan** = Small difference (25%)
- 🟢 **Green** = Medium difference (50%)
- 🟡 **Yellow** = Large difference (75%)
- 🔴 **Red** = Maximum difference (100%)

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| F8  | Toggle 3-panel difference view |
| F9  | Toggle validation reference only |
| F10 | Show validation info |
| F11 | Compare with validation (text) |
| F12 | Promote to validation reference |
| a/d | Previous/next image |
| w/e | First/last image |
| s   | Toggle tailing mode |
| h   | Show help |

## Troubleshooting

### "No validation reference found"
- Make sure validation file exists in `queue/validation/`
- File name should match test name (without [test] suffix)
- Run: `python claude-debug/verify_diff_setup.py`

### "No .npz file found"
- Make sure raw NPZ file exists in `queue/raw/`
- File name should include [test] suffix
- Check: `ls queue/raw/*test*.npz`

### "Shape mismatch"
- Current and validation arrays must have same shape
- Verify with: `python claude-debug/verify_diff_setup.py`

## Implementation Details

### Files Modified
- `yapocis/scripts/imagewatcher.py` - Added difference visualization mode

### Key Functions
- `create_difference_visualization()` - Generates 3-panel view
- `toggle_difference_view()` - Toggles F8 view mode

### Data Flow
1. Load current result from `queue/raw/{test_name} [test].npz`
2. Load validation from `queue/validation/{test_name}.npz`
3. Compute absolute difference
4. Normalize to 0-1 range
5. Apply color mapping (blue→red)
6. Create composite surface with labels
7. Display in pygame window
