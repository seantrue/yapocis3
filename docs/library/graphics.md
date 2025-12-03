# Graphics

2D vector graphics primitives for GPU-accelerated rendering.

## Overview

The graphics library provides anti-aliased 2D drawing primitives. Two variants are available:

- `Graphics` - For 1-channel (grayscale) images
- `Graphics4` - For 4-channel (RGBA) images (recommended)

## Class Reference

```python
from yapocis.lib.graphics4 import Graphics4

g = Graphics4()
```

## Drawing Methods

All methods operate on a canvas array that you provide.

### circles

Draw filled circles with anti-aliasing.

```python
g.circles(canvas, coords, color=[1, 0, 0, 1], distance=50, aa_width=1.5)
```

**Parameters:**
- `canvas` - (H, W, 4) float32 array
- `coords` - List of (x, y) center points
- `color` - RGBA color [0-1]
- `distance` - Circle radius
- `aa_width` - Anti-aliasing width in pixels

### lines

Draw anti-aliased lines.

```python {.no-exec}
g.lines(canvas, coords, color=[1, 1, 1, 1], distance=2.0, aa_width=1.0)
```

**Parameters:**
- `canvas` - (H, W, 4) float32 array
- `coords` - List of (x1, y1, x2, y2) line segments
- `color` - RGBA color
- `distance` - Line width
- `aa_width` - Anti-aliasing width

### ellipses

Draw filled ellipses.

```python {.no-exec}
g.ellipses(canvas, coords, color=[0, 1, 0, 1], aa_width=1.0)
```

**Parameters:**
- `coords` - List of (x, y, rx, ry) center and radii

### composite

Composite pre-rendered blocks onto the canvas.

```python {.no-exec}
g.composite(canvas, locations, blocks)
```

### draw_strings

Render text strings.

```python {.no-exec}
g.draw_strings(canvas, locations, texts, font="default", size=32, color=[1, 1, 1, 1])
```

## Buffer Management

### write

Upload canvas to GPU.

```python
g.write(canvas)
```

### read

Download canvas from GPU.

```python
result = g.read(canvas)
```

## Example

```python
import numpy as np
from yapocis.lib.graphics4 import Graphics4

# Create RGBA canvas
canvas = np.zeros((512, 512, 4), dtype=np.float32)

# Create graphics context
g = Graphics4()

# Draw red circle - coords are (x, y, radius)
g.circles(canvas, [(256, 256, 100)], color=[1, 0, 0, 1])

# Draw green lines - coords are (x1, y1, x2, y2)
line_coords = [
    (100, 100, 400, 100),
    (100, 100, 100, 400),
]
g.lines(canvas, line_coords, color=[0, 1, 0, 1], distance=3.0)

# Read result
result = g.read(canvas)
print(f"Canvas shape: {result.shape}")
```

## Color Parameters

Colors can be specified as:

```python
# Single color for all primitives
color=[1, 0, 0, 1]  # Red, full opacity

# Per-primitive colors
colors=np.array([
    [1, 0, 0, 1],  # First primitive: red
    [0, 1, 0, 1],  # Second primitive: green
    [0, 0, 1, 1],  # Third primitive: blue
], dtype=np.float32)
```

## Width/Distance Parameters

Can be scalar or per-primitive:

```python
# Single width for all
distance=5.0

# Per-primitive widths
widths=np.array([2.0, 5.0, 10.0], dtype=np.float32)
```

## Anti-Aliasing

The `aa_width` parameter controls edge smoothness:

| Value | Effect |
|-------|--------|
| 0.0 | Hard edges (aliased) |
| 1.0 | Standard anti-aliasing |
| 2.0+ | Softer/blurrier edges |

## Resident Mode

For repeated drawing, use resident mode to avoid buffer transfers:

```python {.no-exec}
g.write(canvas)
for frame in range(100):
    g.kmgr.circles_res(canvas, coords, aa_widths, colors)  # _res variant
result = g.read(canvas)
```

## Underlying Kernel: graphics4

The `Graphics4` class uses the `graphics4` interface.

### Interface Definition

```
interface graphics4 {
    kernel lines(width_of a, height_of a, inout float a, width_of lines,
                 in float lines, in float distance, in float aa_width, in float rcolor);
    extension res as lines(..., resident float a, ...);

    kernel circles(width_of a, height_of a, inout float a, width_of lines,
                   in float lines, in float aa_width, in float rcolor);
    extension res as circles(...);

    kernel rects(width_of a, height_of a, inout float a, width_of lines,
                 in float lines, in float aa_width, in float rcolor);
    extension res as rects(...);

    kernel triangles(...);
    kernel ellipses(...);
    kernel composite(...);
};
```

### Kernel Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `width` | `int` | Canvas width (derived from `width_of a`) |
| `height` | `int` | Canvas height (derived from `height_of a`) |
| `a` | `float*` | Canvas buffer (H, W, 4) RGBA |
| `n_primitives` | `int` | Number of primitives (derived from `width_of lines`) |
| `lines` | `float*` | Primitive coordinates |
| `distance` | `float*` | Line width / circle radius |
| `aa_width` | `float*` | Anti-aliasing width |
| `rcolor` | `float*` | RGBA colors (4 floats per primitive) |

### Kernel Variants

| Kernel | Description |
|--------|-------------|
| `lines` | Draw anti-aliased lines |
| `circles` | Draw filled circles |
| `rects` | Draw filled rectangles |
| `triangles` | Draw filled triangles |
| `ellipses` | Draw filled ellipses |
| `composite` | Composite pre-rendered blocks |
| `*_res` variants | Resident buffer versions |

### Direct Kernel Access

```python
from yapocis.core import load_program
import numpy as np

# Load the graphics4 interface
prog = load_program("graphics4")

# Prepare data
canvas = np.zeros((512, 512, 4), dtype=np.float32)
coords = np.array([[100, 100, 400, 400]], dtype=np.float32)  # x1,y1,x2,y2
distance = np.array([3.0], dtype=np.float32)
aa_width = np.array([1.0], dtype=np.float32)
color = np.array([[1, 0, 0, 1]], dtype=np.float32)  # RGBA

# Call kernel directly (positional args)
prog.lines(canvas, coords, distance, aa_width, color)
```

## See Also

- {doc}`kerneldraw` - Lower-level drawing kernels
- {doc}`typeset` - Text rendering utilities
