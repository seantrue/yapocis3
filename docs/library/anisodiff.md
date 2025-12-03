# Anisodiff

Perona-Malik anisotropic diffusion for edge-preserving smoothing.

## Overview

Anisotropic diffusion smooths images while preserving edges. Unlike Gaussian blur which blurs uniformly, anisotropic diffusion reduces smoothing near edges, maintaining image structure.

## Class Reference

```python
from yapocis.lib.anisodiff import Anisodiff

ad = Anisodiff()
```

## Methods

### anisodiff_n

Perform n iterations of Perona-Malik filtering.

```python
result = ad.anisodiff_n(image, n=10, scale=3.0, step_size=0.2)
```

**Parameters:**
- `image` - 2D or 3D float32 array
- `n` - Number of iterations
- `scale` - Edge sensitivity (higher = more smoothing near edges)
- `step_size` - Diffusion step size (must be <= 0.25 for stability)

**Returns:** Filtered image array

### anisodiff

Iterator that yields filtered images indefinitely.

```python {.no-exec}
for i, filtered in enumerate(ad.anisodiff(image, scale=3.0)):
    if i >= 10:
        break
    show_array(filtered, f"iteration_{i}")
```

### anisodiff_rgbI_n

Filter RGB image by extracting and filtering the intensity component.

```python
from yapocis.lib.anisodiff import Anisodiff
from yapocis.utils.basic import get_test_image

rgb_image = get_test_image()
ad = Anisodiff()
result = ad.anisodiff_rgbI_n(rgb_image, n=10, scale=3.0, step_size=0.2)
print(f"Filtered RGB image shape: {result.shape}")
```

### anisodiff_rgbIS_n

Filter RGB image by filtering both intensity and saturation components.

```python
from yapocis.lib.anisodiff import Anisodiff
from yapocis.utils.basic import get_test_image

rgb_image = get_test_image()
ad = Anisodiff()
result = ad.anisodiff_rgbIS_n(rgb_image, n=10, scale=3.0, step_size=0.2)
print(f"Result shape: {result.shape}")
```

## Example

```python
import numpy as np
from yapocis.lib.anisodiff import Anisodiff
from yapocis.utils.basic import get_test_image

# Load image
img = get_test_image()

# Create filter
ad = Anisodiff()

# Apply 20 iterations
filtered = ad.anisodiff_n(img, n=20, scale=3.0, step_size=0.2)
```

## Parameters Guide

| Parameter | Range | Effect |
|-----------|-------|--------|
| `scale` | 1-10 | Lower = more edge preservation |
| `step_size` | 0.1-0.25 | Lower = more stable, slower |
| `n` | 5-50 | More iterations = smoother |

## Algorithm

The Perona-Malik equation:

```
I_t = div(c(x,y,t) * grad(I))
```

Where `c(x,y,t)` is a conductance function that decreases near edges:

```
c(grad(I)) = exp(-(|grad(I)|/K)^2)
```

K is the `scale` parameter.

## Underlying Kernel: anisodiff

The `Anisodiff` class uses the `anisodiff` interface.

### Interface Definition

```
interface anisodiff {
    // HxW image (1 channel)
    kernel filterImage(width_of input, height_of input, in float input,
                       float scale, float size_step, out_like input);
    alias filterImage_res as filterImage(width_of input, height_of input,
                       resident float input, float scale, float size_step,
                       resident float output);

    // HxWx2 image (2 channels)
    kernel filterImage2(...);
    alias filterImage2_res ...;

    // HxWx3 image (3 channels)
    kernel filterImage3(...);
    alias filterImage3_res ...;

    // HxWx4 image (4 channels)
    kernel filterImage4(...);
    alias filterImage4_res ...;
};
```

### Kernel Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `width` | `int` | Image width (derived from `width_of input`) |
| `height` | `int` | Image height (derived from `height_of input`) |
| `input` | `float*` | Input image buffer |
| `scale` | `float` | Edge sensitivity K (higher = more smoothing near edges) |
| `size_step` | `float` | Diffusion step size (≤0.25 for stability) |
| `output` | `float*` | Output filtered image |

### Kernel Variants

| Kernel | Channels | Description |
|--------|----------|-------------|
| `filterImage` | 1 | Grayscale images |
| `filterImage2` | 2 | 2-channel images |
| `filterImage3` | 3 | RGB images |
| `filterImage4` | 4 | RGBA images |
| `*_res` variants | - | Resident buffer versions |

### Direct Kernel Access

```python
from yapocis.core import load_program
from yapocis.utils.basic import get_test_image
import numpy as np

# Load the anisodiff interface
prog = load_program("anisodiff")

# Load test image
rgb_image = get_test_image()
image = np.mean(rgb_image, axis=2).astype(np.float32)

# Call kernel directly (positional args, single iteration)
output = prog.filterImage(image, 3.0, 0.2)  # input, scale, size_step
output3 = prog.filterImage3(rgb_image, 3.0, 0.2)  # 3-channel variant
print(f"Output shapes: {output.shape}, {output3.shape}")
```

## See Also

- {doc}`gaussian` - Uniform Gaussian smoothing
- {doc}`median` - Non-linear noise removal
