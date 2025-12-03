# Gradient

Image gradient magnitude and direction computation.

## Overview

The `Gradient` class computes spatial derivatives of images, returning both the gradient magnitude (edge strength) and direction (edge orientation).

## Class Reference

```python
from yapocis.lib.gradient import Gradient

grad = Gradient()
```

## Methods

### gradient

Compute gradient magnitude and direction.

```python
magnitude, direction = grad.gradient(image, stride=3)
```

**Parameters:**
- `image` - 2D float32 array
- `stride` - Neighbor distance for derivative (default: 3)

**Returns:**
- `magnitude` - Gradient magnitude (edge strength)
- `direction` - Gradient direction in range [-1, 1] (normalized by pi)

## Example

```python
import numpy as np
from yapocis.lib.gradient import Gradient
from yapocis.utils.basic import get_test_image

# Load and convert to grayscale
img = get_test_image()
gray = np.mean(img, axis=2).astype(np.float32)

# Compute gradient
grad = Gradient()
magnitude, direction = grad.gradient(gray, stride=3)

# magnitude: high values = strong edges
# direction: -1 to 1 representing angle/pi
```

## Algorithm

The gradient is computed as:

```
dx = image[x + stride, y] - image[x - stride, y]
dy = image[x, y + stride] - image[x, y - stride]

magnitude = sqrt(dx^2 + dy^2)
direction = atan2(dy, dx) / pi
```

## Stride Parameter

| Stride | Effect |
|--------|--------|
| 1 | Fine edges, noise sensitive |
| 3 | Default, balanced |
| 5+ | Coarser edges, noise resistant |

## Multi-Channel Support

The kernel supports 1, 2, 3, and 4 channel images:

```python {.no-exec}
# Grayscale
grad.gradient(gray_image)

# RGB (computes per-channel, returns per-channel)
grad.gradient(rgb_image)

# RGBA
grad.gradient(rgba_image)
```

## Direction Interpretation

| Value | Direction |
|-------|-----------|
| 0 | Right (positive X) |
| 0.5 | Up (positive Y) |
| 1.0 / -1.0 | Left (negative X) |
| -0.5 | Down (negative Y) |

## Edge Detection Pipeline

Gradient is often used in edge detection:

```python
from yapocis.lib.gradient import Gradient
from yapocis.lib.gaussian import Gaussian

g = Gaussian()
grad = Gradient()

# 1. Blur to reduce noise
smoothed = g.gauss_image(image, scale=2)

# 2. Compute gradient
magnitude, direction = grad.gradient(smoothed)

# 3. Threshold for edges
edges = magnitude > threshold
```

## Underlying Kernel: gradient

The `Gradient` class uses the `gradient` interface.

### Interface Definition

```
interface gradient {
    kernel gradient(width_of input, height_of input, in float input, int reach,
                    out_like input, out_like input);
    extension res of gradient(width_of input, height_of input, resident float input,
                              int reach, resident float grad, resident float theta);
};
```

### Kernel Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `width` | `int` | Image width (derived from `width_of input`) |
| `height` | `int` | Image height (derived from `height_of input`) |
| `input` | `float*` | Input image buffer |
| `reach` | `int` | Neighbor distance for derivative calculation |
| `grad` | `float*` | Output gradient magnitude |
| `theta` | `float*` | Output gradient direction |

### Kernel Variants

| Kernel | Description |
|--------|-------------|
| `gradient` | Standard version with automatic buffer management |
| `gradient_res` | Resident buffer variant for repeated operations |

### Direct Kernel Access

```python
from yapocis.core import load_program
from yapocis.utils.basic import get_test_image
import numpy as np

# Load test image
img = get_test_image()
gray = np.mean(img, axis=2).astype(np.float32)

# Load the gradient interface
prog = load_program("gradient")

# Call kernel directly (positional args)
grad_mag, theta = prog.gradient(gray, 3)  # input, reach -> returns (grad, theta)
print(f"Gradient magnitude shape: {grad_mag.shape}")
print(f"Theta (direction) shape: {theta.shape}")
```

## See Also

- {doc}`gaussian` - Pre-smoothing before gradient
- {doc}`zc` - Zero-crossing edge detection
- {doc}`anisodiff` - Edge-preserving smoothing
