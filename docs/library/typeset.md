# Typeset

Text rendering with FreeType integration.

## Overview

The `typeset` module provides high-quality text rendering using FreeType fonts. It supports TrueType/OpenType fonts with rotation, sizing, and proper kerning.

## Functions

### render_text

Render text to a grayscale image.

```python
from yapocis.lib.typeset import render_text

bbox, image = render_text(
    "Hello World",
    font_name="default",
    size=32,
    rotation=None,
    debug_bbox=False,
    crop=True
)
print(f"Bounding box: {bbox}")
print(f"Image shape: {image.shape}")
```

**Parameters:**
- `text` - String to render
- `font_name` - Font name or "default" (default: "default")
- `size` - Font size in points (default: 32)
- `rotation` - Text angle in degrees (default: None = 0)
- `debug_bbox` - Draw bounding box outline (default: False)
- `crop` - Crop to content bounds (default: True)

**Returns:** (bbox, image) tuple where:
- `bbox` - [x1, y1, x2, y2] bounding box
- `image` - (H, W) float32 array in [0.0, 1.0]

### extent_freetype

Get text extent without rendering.

```python
from yapocis.lib.typeset import extent_freetype
from yapocis.fonts import font

font_path = font("default")
xmin, ymin, xmax, ymax = extent_freetype("Hello", font_path, size=12, angle=0)
print(f"Text extent: ({xmin}, {ymin}) to ({xmax}, {ymax})")
```

**Parameters:**
- `text` - String to measure
- `filename` - Path to font file
- `size` - Font size in points
- `angle` - Rotation angle in degrees

**Returns:** (xmin, ymin, xmax, ymax) bounds

### render_freetype

Low-level rendering to uint8 array.

```python
from yapocis.lib.typeset import render_freetype
from yapocis.fonts import font

font_path = font("default")
image = render_freetype("Test", font_path, size=12, angle=0)
print(f"Rendered image shape: {image.shape}")
```

**Returns:** (H, W) uint8 array

## Usage

### Basic Text Rendering

```python
from yapocis.lib.typeset import render_text

bbox, text_img = render_text("Hello World", size=48)

# Result is grayscale float32 [0, 1]
print(f"Size: {text_img.shape}")
```

### Rotated Text

```python
bbox, rotated = render_text("Rotated", size=32, rotation=45)
```

### Custom Font

```python
from yapocis.fonts import font

font_path = font("arial")  # Get font path from yapocis.fonts
bbox, text_img = render_text("Custom Font", font_name="arial", size=24)
```

### Integration with Graphics

```python {.no-exec}
import numpy as np
from yapocis.lib.graphics4 import Graphics4
from yapocis.lib.typeset import render_text

# Create canvas
canvas = np.zeros((256, 512, 4), dtype=np.float32)

# Render text
bbox, text_img = render_text("GPU Text", size=48)

# Composite onto canvas using Graphics4
g = Graphics4()
g.composite(canvas, [(100, 100)], [text_img], alpha=1.0)
result = g.read(canvas)
```

## Caching

The module uses three caches for performance:

- `_glyph_cache` - Rendered glyph bitmaps
- `_kerning_cache` - Kerning values between character pairs
- `_extent_cache` - Pre-computed text extents

Caches are indexed by (font, size, angle) tuples. Changing fonts or sizes frequently will reduce cache effectiveness.

## Rotation Handling

Text rotation affects layout computation:

| Angle Range | Layout Adjustment |
|-------------|-------------------|
| 0° | Horizontal baseline |
| 0° to 90° | Shift down |
| 90° | Vertical, shift left |
| 90° to 180° | Shift left and down |
| -90° to 0° | No adjustment |
| -180° to -90° | Shift left |

## Font Resolution

Fonts are resolved through `yapocis.fonts.font()`:

```python
from yapocis.fonts import font

# Standard fonts
arial = font("arial")
default = font("default")

# Custom path
custom = font("/path/to/font.ttf")
```

## Notes

- Uses FreeType via `freetype-py` library
- Supports TrueType (.ttf) and OpenType (.otf) fonts
- Angles are in degrees, counter-clockwise
- Output is anti-aliased grayscale

## See Also

- {doc}`graphics` - Drawing primitives with text support
- {doc}`kerneldraw` - Display list rendering with text
