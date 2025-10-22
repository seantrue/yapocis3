# Difference Visualization Mode for Imagewatcher

## Overview

Added a new 3-panel difference visualization mode to imagewatcher that displays:
1. **Current test image** (left panel)
2. **Validation reference** (middle panel)
3. **Color-coded difference map** (right panel)

This makes it easy to see where and how large differences are between test results and validation references.

## Features

### Heat Map Visualization
The difference map uses an intuitive color gradient to show difference magnitude:
- **Blue** (0.0) - No difference
- **Cyan** (0.25) - Small difference
- **Green** (0.50) - Medium difference
- **Yellow** (0.75) - Large difference
- **Red** (1.0) - Maximum difference

### Color Scale Legend
Each difference view includes a color scale legend showing:
- Gradient bar from blue to red
- Minimum value (0)
- Maximum absolute difference value

### Multi-Format Support
Works with various image formats:
- **Grayscale** (2D arrays)
- **RGB** (3-channel)
- **RGBA** (4-channel, displays RGB portion)
- **Float32** (normalized to 0-1 or auto-scaled)
- **Multi-channel** (uses max difference across channels)

## Usage

### Keyboard Shortcut
Press **F8** to toggle the 3-panel difference view on/off

### Command Line
```bash
yapocis-imagewatcher --queue <queue_directory>
# Navigate to an image with validation reference
# Press F8 to view differences
```

### Navigation
- **F8** - Toggle 3-panel difference view
- **F9** - Toggle between test image and validation reference only
- **a/d** - Previous/next image (auto-exits difference view)
- **w/e** - First/last image (auto-exits difference view)

## Implementation Details

### File: `yapocis/scripts/imagewatcher.py`

#### New Functions

**`create_difference_visualization(font)`** (lines 867-1096)
- Loads current test result and validation reference from .npz files
- Validates shapes match
- Normalizes arrays to uint8 for display
- Computes absolute difference
- Creates color-coded heat map using 5-stop gradient
- Builds 3-panel composite surface with labels
- Adds color scale legend with min/max labels

**`toggle_difference_view(font)`** (lines 836-857)
- Toggles between test image and difference view
- Disables validation-only view when active
- Shows status messages

#### Modified Functions

**Navigation functions** (lines 514-545)
- `next_image()`, `previous_image()`, `first_image()`, `last_image()`
- Now reset `showing_difference_view` flag when navigating

**`step_image()`** (lines 548-572)
- Resets difference view in tailing mode

**`toggle_validation_view()`** (lines 812-833)
- Disables difference view when switching to validation-only view

#### Global State
- Added `showing_difference_view` flag (line 162)
- Tracks whether difference visualization is active

#### Keyboard Help
- Updated help text with F8 shortcut (line 62)

#### Event Handler
- Added F8 key handler (lines 1352-1354)

## Testing

### Test Script: `claude-debug/test_diff_view.py`

Creates 4 test scenarios:

1. **Grayscale with small differences**
   - Gradient with Gaussian noise (σ=5)
   - Tests grayscale visualization

2. **RGB with localized differences**
   - Colored squares with position and color changes
   - Tests RGB multi-channel handling

3. **Float32 with tiny differences**
   - Random noise with very small variance (σ=0.001)
   - Tests float normalization and sensitivity

4. **RGBA with alpha channel differences**
   - Semi-transparent gradient with alpha modifications
   - Tests RGBA handling

### Running Tests
```bash
# Generate test images
python claude-debug/test_diff_view.py

# View in imagewatcher
yapocis-imagewatcher --queue claude-debug/test_diff_queue

# Press F8 to toggle difference view
```

## Technical Notes

### Difference Computation
- Uses absolute difference: `|current - reference|`
- For multi-channel images, takes max difference across channels
- Normalizes to 0-1 range for visualization

### Color Mapping Algorithm
- Piecewise linear interpolation through 5 color stops
- Blue (0.0) → Cyan (0.25) → Green (0.5) → Yellow (0.75) → Red (1.0)
- Implemented with conditional branches for efficiency

### Layout
- 3 equal-width panels side by side
- Each panel labeled at top
- Color scale legend below difference map
- Automatic scaling to fit window (WIDTH // 3 - 10 per panel)

### Performance Considerations
- Uses PIL for high-quality image resizing (LANCZOS)
- Difference computation in vectorized NumPy
- Color mapping uses explicit loops (consider vectorization for very large images)

## Benefits

1. **Immediate visual feedback** - See exactly where differences occur
2. **Magnitude awareness** - Color coding shows how large differences are
3. **Side-by-side comparison** - All three views visible simultaneously
4. **Quantitative info** - Max difference value displayed in label and legend
5. **Format agnostic** - Works with any numeric array format

## Future Enhancements (Optional)

- Vectorize color mapping for better performance
- Add adjustable difference threshold/sensitivity
- Export difference visualization to file
- Overlay mode (current with difference highlighted)
- Per-channel difference views for RGB
- Histogram of difference magnitudes
