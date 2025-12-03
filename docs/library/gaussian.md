# Gaussian

Gaussian blur and Difference of Gaussians (DoG) edge detection.

## Overview

The `Gaussian` class provides separable Gaussian convolution for efficient blurring at multiple scales, plus DoG-based edge detection with zero-crossing extraction.

## Class Reference

```python
from yapocis.lib.gaussian import Gaussian

g = Gaussian()
```

## Constructor Options

```python
g = Gaussian(
    device_name=None,           # Specific device
    gaussian_weights=None,      # Custom weight generator
    use_async=True,             # Async execution (default)
)
```

## Methods

### gauss_image

Apply Gaussian blur at a specific scale.

```python
blurred = g.gauss_image(image, scale=3)
```

**Parameters:**
- `image` - 2D float32 array
- `scale` - Scale index (0 = smallest blur, higher = more blur)

**Returns:** Blurred image

### zcsdog

Compute Difference of Gaussians with zero-crossing detection.

```python
zc, grad, theta = g.zcsdog(image, scale=2)
```

**Parameters:**
- `image` - 2D float32 array
- `scale` - Scale index
- `threshold` - Zero-crossing threshold (default: 7.5e-06)
- `clear_margin` - Zero out edges (default: True)
- `frame` - Add frame markers (default: True)

**Returns:**
- `zc` - Zero-crossing image (edges)
- `grad` - Gradient magnitude
- `theta` - Gradient direction

### get_scales

Get the number of available blur scales.

```python
n = g.get_scales()  # e.g., 15
```

### get_gaussian_kernel

Get a specific Gaussian kernel object.

```python
kernel = g.get_gaussian_kernel(scale=3)
```

## Example

```python
import numpy as np
from yapocis.lib.gaussian import Gaussian
from yapocis.utils.basic import get_test_image

# Load image
img = get_test_image()
gray = np.mean(img, axis=2).astype(np.float32)  # Convert to grayscale

# Create Gaussian processor
g = Gaussian()
print(f"Available scales: {g.get_scales()}")  # 0 to n-1

# Apply blur at different scales (within available range)
blur_small = g.gauss_image(gray, scale=0)
blur_medium = g.gauss_image(gray, scale=3)
blur_large = g.gauss_image(gray, scale=6)
print(f"Blur shapes: small={blur_small.shape}, medium={blur_medium.shape}, large={blur_large.shape}")

# Edge detection
edges, gradient, direction = g.zcsdog(gray, scale=2)
print(f"Edge detection: edges={edges.shape}, gradient={gradient.shape}")
```

## Scale Progression

Gaussian kernels are generated with exponentially increasing sigma values. Default parameters from `convolvesep.yml`:

| Scale | Approximate Sigma | Kernel Width |
|-------|-------------------|--------------|
| 0 | 1.0 | 5 |
| 1 | 1.6 | 9 |
| 2 | 2.56 | 13 |
| 3 | 4.1 | 21 |
| ... | ... | ... |

## Underlying Kernel: convolvesep

The `Gaussian` class uses the `convolvesep` interface, which generates separable convolution kernels dynamically based on Gaussian weight parameters.

### Interface Definition

```
interface convolvesep {
    kernel ${name}(int vertical, width_of input, height_of input, in float input, out_like input);
    extension res of ${name}(int vertical, width_of input, height_of input, resident float input, resident float output);
    kernel ${name}_h(width_of input, height_of input, resident float input, resident float output);
    kernel ${name}_v(width_of input, height_of input, resident float input, resident float output);
};
```

This is a templated interface—the `${name}` is replaced at generation time with kernel names like `gauss5`, `gauss9`, `gauss13`, etc.

### Generated Kernels

For each Gaussian scale, four kernel variants are generated:

| Kernel | Parameters | Description |
|--------|------------|-------------|
| `gauss{N}` | `vertical, width, height, input, output` | Generic convolution (vertical=0 for horizontal, 1 for vertical) |
| `gauss{N}_h` | `width, height, input, output` | Horizontal convolution only |
| `gauss{N}_v` | `width, height, input, output` | Vertical convolution only |
| `gauss{N}_res` | `vertical, width, height, input, output` | Resident buffer variant (no transfers) |

Where `N` is the kernel width (e.g., 5, 9, 13, 21, ...).

### Kernel Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `vertical` | `int` | 0 = horizontal pass, 1 = vertical pass |
| `width` | `int` | Image width (derived from `width_of input`) |
| `height` | `int` | Image height (derived from `height_of input`) |
| `input` | `float*` | Input image buffer |
| `output` | `float*` | Output image buffer |

### Weight Generation

Gaussian weights are generated from parameters in `convolvesep.yml`:

```yaml
gaussian_params:
  max_width: 150    # Maximum kernel size
  scale: 1.6        # Sigma progression factor
  limit: 0.001      # Weight truncation threshold
```

The `GaussianWeights` class generates kernels with exponentially increasing sigma:

```python
# sigma values: 1.0, 1.6, 2.56, 4.096, 6.55, ...
# kernel sizes: 5, 9, 13, 21, 33, ...
```

### Direct Kernel Access

```python
from yapocis.core import load_program
from yapocis.utils.basic import get_test_image
import numpy as np

# Load test image
img = get_test_image()
gray = np.mean(img, axis=2).astype(np.float32)

# Load the convolvesep interface
prog = load_program("convolvesep")

# Call kernel directly (positional args)
# Available kernels: gauss9, gauss13, gauss19, gauss27, gauss39, gauss59, gauss87, gauss127
output = prog.gauss9(0, gray)  # vertical=0 for horizontal pass
output = prog.gauss27(1, gray)  # vertical=1 for vertical pass
print(f"Gaussian output shape: {output.shape}")
```

### Custom Weight Configuration

Override default Gaussian parameters:

```python {.no-exec}
from yapocis.lib.gaussian import Gaussian, GaussianWeights

# Custom weight generator
gw = GaussianWeights(
    max_width=200,    # Larger kernels
    scale=1.5,        # Finer scale progression
    limit=0.0001      # More accurate (slower)
)

g = Gaussian(gaussian_weights=gw)
```

Or use specific sigma values:

```python {.no-exec}
gw = GaussianWeights(
    max_width=200,
    sigmas=[1.0, 2.0, 4.0, 8.0, 16.0],  # Exactly these scales
    limit=0.001
)
```

## Separable Convolution

Uses separable 1D convolutions (horizontal then vertical) for efficiency:

```
O(n * k)  vs  O(n * k^2)  for 2D direct convolution
```

Where n = image pixels, k = kernel width.

## See Also

- {doc}`anisodiff` - Edge-preserving smoothing
- {doc}`gradient` - Gradient computation
- {doc}`zc` - Zero-crossing utilities
