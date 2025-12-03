# KernelDraw

GPU drawing with display list abstraction.

## Overview

The `KernelDraw` classes provide a display list API for efficient GPU drawing. Commands are queued and batch-rendered for optimal performance.

Covers `kerneldraw1.py` (grayscale), `kerneldraw3.py` (RGB), and `kerneldraw4.py` (RGBA).

## Classes

| Class | Channels | Description |
|-------|----------|-------------|
| `KernelDraw1` | 1 | Grayscale images |
| `KernelDraw3` | 3 | RGB images |
| `KernelDraw4` | 4 | RGBA images |

## Class Reference

```python
from yapocis.lib.kerneldraw4 import KernelDraw4

canvas = np.zeros((512, 512, 4), dtype=np.float32)
kd = KernelDraw4(canvas)
```

## Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | ndarray | Target image array |
| `device_name` | str | Optional GPU device name |
| `graphics` | class | Optional Graphics class override |
| `chunk_size` | int | Batch size for rendering (default: 500) |

## Display List Commands

### Individual Primitives

Add single primitives to the display list:

```python
# Lines (x1, y1, x2, y2)
kd.line((10, 10, 100, 100), color=(1, 0, 0, 1), width=2)

# Rectangles (x1, y1, x2, y2)
kd.rect((50, 50, 150, 150), color=(0, 1, 0, 1))

# Circles (cx, cy, radius)
kd.circle((100, 100, 50), color=(0, 0, 1, 1))

# Triangles (x1, y1, x2, y2, x3, y3)
kd.triangle((100, 50, 50, 150, 150, 150), color=(1, 1, 0, 1))

# Ellipses (cx, cy, rx, ry)
kd.ellipse((100, 100, 80, 40), color=(1, 0, 1, 1))
```

### Text

```python
kd.draw_string((100, 100), "Hello", font="default", size=24, color=(1, 1, 1, 1))
```

### Sprites

```python {.no-exec}
kd.sprite((x, y), sprite_image, alpha=1.0)
```

## Batch Rendering

For efficiency, add multiple primitives then render all at once:

```python {.no-exec}
# Add many primitives
for i in range(100):
    kd.circle((i * 5, i * 3, 10), color=(1, 0, 0, 1))

for i in range(50):
    kd.line((0, i * 10, 500, i * 10), width=1)

# Render all at once
result = kd.read()  # Renders and downloads
```

## Batch Drawing Methods

Draw multiple primitives at once:

```python
# Multiple lines with arrays
coords = [(10, 10, 100, 100), (20, 20, 110, 110), (30, 30, 120, 120)]
kd.lines(coords, color=(1, 0, 0, 1))

# With per-primitive colors
colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
kd.lines(coords, colors=colors)

# Multiple circles
circles = [(100, 100, 20), (200, 150, 30), (150, 200, 25)]
kd.circles(circles, color=(0, 1, 0, 1))
```

## Rendering Control

### Manual Render

```python {.no-exec}
# Render specific primitive types
kd.render_lines()
kd.render_circles()
kd.render_rects()
kd.render_triangles()
kd.render_ellipses()
kd.render_draw_strings()
kd.render_sprites()

# Render all
kd.render()
```

### Read Result

```python {.no-exec}
# Render and download
result = kd.read()

# Just snapshot (keep on GPU)
kd.snapshot()
```

## Properties

| Property | Description |
|----------|-------------|
| `alpha` | Default alpha for new primitives |
| `color` | Default color for new primitives |
| `chunk_size` | Batch size for kernel calls |

## Example: Visualization

```python {.no-exec}
import numpy as np
from yapocis.lib.kerneldraw4 import KernelDraw4

# Create canvas
canvas = np.zeros((512, 512, 4), dtype=np.float32)
kd = KernelDraw4(canvas)

# Background gradient
for y in range(512):
    kd.line((0, y, 512, y), color=(0, 0, y/512, 1))

# Draw data points
data = np.random.rand(100, 2) * 400 + 50
for x, y in data:
    kd.circle((x, y, 5), color=(1, 0.5, 0, 0.8))

# Axes
kd.line((50, 50, 50, 450), color=(1, 1, 1, 1), width=2)
kd.line((50, 450, 450, 450), color=(1, 1, 1, 1), width=2)

# Labels
kd.draw_string((250, 480), "X Axis", size=20, color=(1, 1, 1, 1))
kd.draw_string((20, 250), "Y", size=20, color=(1, 1, 1, 1), rotation=90)

# Render
result = kd.read()
```

## Async Mode

Enable async execution via environment variable:

```bash
export YAPOCIS_ASYNC=1
```

```python
# Async mode wraps render in async context
kd = KernelDraw4(canvas)
# ... add primitives ...
result = kd.read()  # Uses async if enabled
```

## See Also

- {doc}`graphics` - Lower-level Graphics4 class
- {doc}`typeset` - Text rendering
