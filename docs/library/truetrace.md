# TrueTrace

Bitmap to vector contour tracing using GPU-accelerated signed distance fields.

## Overview

TrueTrace converts binary masks to vector contours using:
1. Jump Flood Algorithm (JFA) for Euclidean distance transform
2. Marching Squares for iso-contour extraction
3. Douglas-Peucker simplification
4. Region nesting for outer/hole detection

## Class Reference

```python
from yapocis.lib.truetrace import TrueTrace

tt = TrueTrace()
```

## GPU Methods

### trace_segments

Extract line segments at a given iso level on the signed distance field.

```python {.no-exec}
segments, (width, height) = tt.trace_segments(mask, iso=0.0)
```

**Parameters:**
- `mask` - HxW uint8 array (0/1 or 0/255)
- `iso` - Distance from edge (0.0 = exact edge, positive = outward, negative = inward)

**Returns:**
- `segments` - Nx4 float32 array of [x0, y0, x1, y1] line segments
- `size` - (width, height) tuple

## CPU Helper Functions

### load_mask_from_image

Load an image and create a binary mask.

```python
from yapocis.lib.truetrace import load_mask_from_image
from yapocis.images import find_image

# Load mask from a project image
mask = load_mask_from_image(find_image("test"), invert=False, threshold=None)
```

**Parameters:**
- `path` - Image file path
- `invert` - Flip foreground/background
- `threshold` - Manual threshold (None = Otsu automatic)

### trace_bitmap

Extract closed contour rings from a binary mask (CPU-only).

```python {.no-exec}
from yapocis.lib.truetrace import trace_bitmap

rings = trace_bitmap(mask, simplify_tol_px=0.75, min_area_px2=4.0)
```

**Returns:** List of closed polyline arrays

### douglas_peucker

Simplify a polyline using Douglas-Peucker algorithm.

```python {.no-exec}
from yapocis.lib.truetrace import douglas_peucker

simplified = douglas_peucker(polyline, tol=1.0)
```

### chain_segments_to_polylines

Chain disconnected segments into continuous polylines.

```python {.no-exec}
from yapocis.lib.truetrace import chain_segments_to_polylines

polylines = chain_segments_to_polylines(segments, tol=0.25)
```

### build_regions_with_holes

Group closed rings into nested regions with hole detection.

```python {.no-exec}
from yapocis.lib.truetrace import build_regions_with_holes

regions = build_regions_with_holes(rings)
# Returns: [{'outer': ring, 'holes': [ring, ...]}, ...]
```

### write_svg_regions

Export regions to SVG file.

```python {.no-exec}
from yapocis.lib.truetrace import write_svg_regions

write_svg_regions("output.svg", regions, w, h,
                  stroke_width=1.0, fill="black", curve="none")
```

## Example: Complete Pipeline

```python {.no-exec}
import numpy as np
from yapocis.lib.truetrace import (
    TrueTrace, load_mask_from_image, chain_segments_to_polylines,
    douglas_peucker, close_and_filter_polylines, build_regions_with_holes,
    write_svg_regions
)
from yapocis.images import find_image

# Load binary mask from project images
mask = load_mask_from_image(find_image("test"), invert=True)

# GPU: Extract segments at edge
tt = TrueTrace()
segments, (w, h) = tt.trace_segments(mask, iso=0.0)

# CPU: Chain into polylines
polylines = chain_segments_to_polylines(segments, tol=0.25)

# CPU: Simplify
simplified = [douglas_peucker(p, tol=0.75) for p in polylines]

# CPU: Close and filter small shapes
rings = close_and_filter_polylines(simplified, min_area_px2=4.0)

# CPU: Detect holes
regions = build_regions_with_holes(rings)

# Export to SVG
write_svg_regions("output.svg", regions, w, h, fill="black")
```

## ISO Parameter

The `iso` parameter controls the contour offset from the mask edge:

| Value | Effect |
|-------|--------|
| 0.0 | Exact edge of mask |
| 1.0 | 1 pixel outward |
| -1.0 | 1 pixel inward |
| 2.0 | 2 pixels outward (creates outline) |

## Simplification Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `simplify_tol_px` | 0.75 | Douglas-Peucker tolerance |
| `min_area_px2` | 4.0 | Discard shapes smaller than this |
| `chain_tol_px` | 0.25 | Endpoint snapping for chaining |

## CPU-Only Mode

For CPU-only tracing without GPU:

```python {.no-exec}
from yapocis.lib.truetrace import (
    sdf_from_mask_cpu, marching_squares_cpu, trace_bitmap
)

# Using convenience function
rings = trace_bitmap(mask)

# Or manual pipeline
sdf = sdf_from_mask_cpu(mask)
segments = marching_squares_cpu(sdf, iso=0.0)
```

## Underlying Kernel: truetrace

The `TrueTrace` class uses the `truetrace` interface.

### Interface Definition

```
interface truetrace {
    # Initialize JFA seeds and distance from binary mask
    kernel init_seed_from_mask(width_of mask, height_of mask,
                               in int mask, out float seed, out float dist);

    # One JFA step with given jump length
    kernel jfa_step(width_of seed_in, height_of seed_in, int jump,
                    in float seed_in, inout float dist, out float seed_out);

    # Marching squares to extract line segments at iso level
    kernel marching_squares(width_of dist, height_of dist, in float dist,
                            float iso, inout uint32 seg_counter,
                            out float segments_out);
};
```

### Kernel Parameters

**init_seed_from_mask:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `width/height` | `int` | Image dimensions (derived) |
| `mask` | `int*` | Binary mask (0/1) |
| `seed` | `float*` | Output seed positions (2 floats per pixel) |
| `dist` | `float*` | Output distance field |

**jfa_step:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `jump` | `int` | Jump distance for this iteration |
| `seed_in` | `float*` | Input seed positions |
| `dist` | `float*` | Distance field (updated in place) |
| `seed_out` | `float*` | Output seed positions |

**marching_squares:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `dist` | `float*` | Signed distance field |
| `iso` | `float` | Iso-contour level |
| `seg_counter` | `uint32*` | Segment count (atomic) |
| `segments_out` | `float*` | Output line segments (x0,y0,x1,y1 per segment) |

### Direct Kernel Access

```python {.no-exec}
from yapocis.core import load_program
import numpy as np

# Load the truetrace interface
prog = load_program("truetrace")

# Initialize from mask
mask = (image > 0.5).astype(np.int32)
seed = np.empty((h, w, 2), dtype=np.float32)
dist = np.empty((h, w), dtype=np.float32)
prog.init_seed_from_mask(mask, seed, dist)

# Run JFA iterations
jump = max(w, h) // 2
while jump >= 1:
    seed_out = np.empty_like(seed)
    prog.jfa_step(jump, seed, dist, seed_out)
    seed = seed_out
    jump //= 2

# Extract contours
counter = np.array([0], dtype=np.uint32)
segments = np.empty((max_segments, 4), dtype=np.float32)
prog.marching_squares(dist, 0.0, counter, segments)
```

## See Also

- {doc}`graphics` - Drawing primitives
- {doc}`imageops` - Image manipulation
