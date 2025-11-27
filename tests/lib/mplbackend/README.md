# Matplotlib Backend Position Validation Tests

## Overview

These tests validate that the yapocis matplotlib backend correctly positions graphics primitives
and images at the expected pixel locations.

## Reference Standard: Agg Backend

**All position validation tests use matplotlib's Agg backend as the reference standard.**

The Agg (Anti-Grain Geometry) backend is matplotlib's standard reference implementation for
high-quality 2D rendering. It is:

- **Widely tested**: The most mature and stable matplotlib backend
- **Platform independent**: Produces identical output across all platforms
- **Well documented**: Serves as the canonical implementation for matplotlib's rendering model

By comparing our yapocis backend output against Agg, we ensure that graphics primitives and
images are positioned correctly according to matplotlib's coordinate system conventions.

## Test Files

### `test_imshow_position.py`
Tests image positioning with `imshow()`:
- `test_imshow_simple_position()` - Single image at specific position
- `test_imshow_position_consistency()` - Corner-positioned images vs Agg
- `test_imshow_single_image_position()` - Multi-quadrant image positioning

### `test_primitives_position.py`
Tests graphics primitives positioning:
- `test_lines_position_consistency()` - Horizontal and vertical lines
- `test_rectangles_position_consistency()` - Filled and outlined rectangles
- `test_circles_position_consistency()` - Filled and outlined circles
- `test_ellipse_position_consistency()` - Ellipses with various rotations
- `test_polygons_position_consistency()` - Triangles, pentagons, hexagons
- `test_text_position_consistency()` - Text with various alignments
- `test_mixed_primitives_position_consistency()` - Complex scenes with multiple primitive types

## Test Methodology

Each test:
1. **Renders with Agg backend** (reference standard)
2. **Renders with yapocis backend** (system under test)
3. **Compares pixel outputs** using statistical metrics:
   - Mean pixel difference (should be < 10-15 on 0-255 scale)
   - Percentage of similar pixels (should be > 90-95%)
   - Maximum difference (informational)

## Coordinate System Notes

### Matplotlib Coordinates
- Origin at **bottom-left** of figure
- Y-axis increases **upward** (y-up)
- Extent for images: `[left, right, bottom, top]`

### Canvas Coordinates (Internal)
- Origin at **top-left** of canvas
- Y-axis increases **downward** (y-down)
- Conversion: `canvas_y = canvas_height - matplotlib_y`

### Position Validation
Tests verify that the coordinate transformations are correct by:
- Placing primitives at known positions in matplotlib coordinates
- Verifying they appear at the correct pixel locations in the output
- Comparing against Agg backend output pixel-by-pixel

## Running Tests

```bash
# Run all position tests
pytest tests/lib/mplbackend/test_primitives_position.py tests/lib/mplbackend/test_imshow_position.py -v

# Run specific test
pytest tests/lib/mplbackend/test_primitives_position.py::test_lines_position_consistency -v

# Run with verbose output
pytest tests/lib/mplbackend/test_primitives_position.py -v -s
```

## Tolerance Values

Different primitives have different tolerance thresholds due to rendering variations:

- **Lines, rectangles, circles, ellipses, polygons**: mean_diff < 10.0, similar pixels > 95%
- **Text rendering**: mean_diff < 15.0, similar pixels > 90% (looser due to font anti-aliasing)
- **Mixed scenes**: mean_diff < 12.0, similar pixels > 95%

These tolerances account for minor anti-aliasing and rounding differences while ensuring
that positioning is functionally correct.
