# Zero Crossing

Zero-crossing detection for edge detection.

## Overview

The `ZC` class detects zero-crossings in images, commonly used after Laplacian or Difference of Gaussians (DoG) filtering to find edges.

## Class Reference

```python
from yapocis.lib.zc import ZC

zc = ZC()
```

## Methods

### zcs

Detect zero-crossings in an image.

```python
edges = zc.zcs(image, threshold=0.00001)
```

**Parameters:**
- `image` - 2D, 3D, or 4D float32 array
- `threshold` - Minimum gradient magnitude at crossing

**Returns:** Binary edge image

### zcs_res

Resident mode version for repeated operations.

```python {.no-exec}
zc.zcs_res(image, output, threshold=0.00001)
```

## Usage

```python
import numpy as np
from yapocis.lib.zc import ZC
from yapocis.lib.gaussian import Gaussian
from yapocis.utils.basic import get_test_image

# Load image and convert to grayscale
img = get_test_image()
gray = np.mean(img, axis=2).astype(np.float32)

# Apply Laplacian/DoG
g = Gaussian()
small = g.gauss_image(gray, scale=2)
large = g.gauss_image(gray, scale=3)
dog = small - large

# Find zero crossings
zc = ZC()
edges = zc.zcs(dog, threshold=0.00001)
```

## Multi-Channel Support

Automatically handles different channel counts:

| Channels | Method Used |
|----------|-------------|
| 1 | `zcs1` |
| 2 | `zcs2` |
| 3 | `zcs3` |
| 4 | `zcs4` |

```python {.no-exec}
# Grayscale
edges = zc.zcs(gray_image)

# RGB
edges_rgb = zc.zcs(rgb_image)

# RGBA
edges_rgba = zc.zcs(rgba_image)
```

## Threshold Parameter

The threshold filters out weak zero crossings:

| Threshold | Effect |
|-----------|--------|
| 0.00001 | Very sensitive (default) |
| 0.001 | Moderate filtering |
| 0.01 | Strong filtering, fewer edges |

## Edge Detection Pipeline

Zero crossings are typically the final step in edge detection:

```python {.no-exec}
from yapocis.lib.gaussian import Gaussian
from yapocis.lib.zc import ZC

g = Gaussian()
zc = ZC()

# Complete edge detection
edges, gradient, direction = g.zcsdog(image, scale=3)
```

Or manually:

```python {.no-exec}
# 1. Smooth
smoothed = g.gauss_image(image, scale=2)

# 2. Compute second derivative (Laplacian approximation via DoG)
large = g.gauss_image(image, scale=3)
dog = smoothed - large

# 3. Find zero crossings
edges = zc.zcs(dog, threshold=0.0001)
```

## Algorithm

A zero crossing occurs where the sign of the image changes between adjacent pixels. The kernel checks 4-neighbors and marks pixels where:
1. A sign change occurs
2. The gradient magnitude exceeds the threshold

## Underlying Kernel: zcs

The `ZC` class uses the `zcs` interface.

### Interface Definition

```
interface zcs {
    kernel zcs1(width_of input, height_of input, float threshold,
                in float input, out_like input);
    alias zcs1_res as zcs1(..., resident float input, resident float zcs);

    kernel zcs2(width_of input, height_of input, float threshold,
                in float input, out_like input);
    alias zcs2_res as zcs2(...);

    kernel zcs3(width_of input, height_of input, float threshold,
                in float input, out_like input);
    alias zcs3_res as zcs3(...);

    kernel zcs4(width_of input, height_of input, float threshold,
                in float input, out_like input);
    alias zcs4_res as zcs4(...);

    kernel sub(in float input1, in float input2, out_like input1);
    alias sub_res as sub(...);
};
```

### Kernel Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `width` | `int` | Image width (derived) |
| `height` | `int` | Image height (derived) |
| `threshold` | `float` | Minimum gradient magnitude |
| `input` | `float*` | Input image |
| `output` | `float*` | Output edge image |

### Kernel Variants

| Kernel | Channels | Description |
|--------|----------|-------------|
| `zcs1` | 1 | Grayscale zero-crossing |
| `zcs2` | 2 | 2-channel zero-crossing |
| `zcs3` | 3 | RGB zero-crossing |
| `zcs4` | 4 | RGBA zero-crossing |
| `sub` | any | Subtraction helper |
| `*_res` variants | - | Resident buffer versions |

### Direct Kernel Access

```python
from yapocis.core import load_program
from yapocis.utils.basic import get_test_image
import numpy as np

# Load test image
img = get_test_image()
gray = np.mean(img, axis=2).astype(np.float32)

# Load the zcs interface
prog = load_program("zcs")

# Call kernel directly (positional args)
edges = prog.zcs1(0.00001, gray)  # threshold, input
print(f"Edges shape: {edges.shape}")
```

## See Also

- {doc}`gaussian` - Gaussian blur and DoG computation
- {doc}`gradient` - Gradient magnitude computation
