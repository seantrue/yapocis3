# Median

3x3 median filter for noise removal.

## Overview

The median filter replaces each pixel with the median of its 3x3 neighborhood. This is effective for removing salt-and-pepper noise while preserving edges.

## Class Reference

```python
from yapocis.lib.median import Median

m = Median()
```

## Methods

### median3x3

Apply 3x3 median filter with optional iterations.

```python
result = m.median3x3(image, iterations=1)
```

**Parameters:**
- `image` - 2D float32 array
- `iterations` - Number of filter passes (default: 1)

**Returns:** Filtered image (copy)

### median3x3fast

Fast version using ping-pong GPU buffers for multiple iterations.

```python
result = m.median3x3fast(image, iterations=5)
```

More efficient for multiple iterations as data stays on GPU.

### median3x3slow

Simple version that transfers data each iteration.

```python
result = m.median3x3slow(image, iterations=1)
```

## Example

```python
import numpy as np
from yapocis.lib.median import Median
from yapocis.utils.basic import get_test_image

# Load image and add salt-and-pepper noise
img = get_test_image()
gray = np.mean(img, axis=2).astype(np.float32)

# Add noise
noisy = gray.copy()
noise_mask = np.random.random(gray.shape) < 0.05
noisy[noise_mask] = np.random.choice([0.0, 1.0], size=noise_mask.sum())

# Apply median filter
m = Median()
filtered = m.median3x3(noisy, iterations=1)
```

## Comparison with Gaussian

| Feature | Median | Gaussian |
|---------|--------|----------|
| Noise type | Salt-and-pepper | Gaussian noise |
| Edge preservation | Excellent | Poor |
| Speed | Moderate | Fast |
| Kernel size | Fixed 3x3 | Variable |

## Multiple Iterations

More iterations provide more smoothing:

| Iterations | Effect |
|------------|--------|
| 1 | Light noise removal |
| 2-3 | Moderate smoothing |
| 5+ | Strong smoothing |

```python {.no-exec}
# Progressive filtering
result1 = m.median3x3(image, iterations=1)
result3 = m.median3x3(image, iterations=3)
result5 = m.median3x3(image, iterations=5)
```

## Async Mode

Enable async execution for better performance:

```python {.no-exec}
m = Median()
m.use_async = True
result = m.median3x3fast(image, iterations=10)
```

## Underlying Kernel: median3x3

The `Median` class uses the `median3x3` interface.

### Interface Definition

```
interface median3x3 {
    kernel median3x3(size_of int a, height_of int a, in float a, out_like a);
    extension res of median3x3(size_of int a, height_of int a, in float a,
                                out float ret);
};
```

### Kernel Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `size` | `int` | Total pixels (derived from `size_of a`) |
| `height` | `int` | Image height (derived from `height_of a`) |
| `a` | `float*` | Input image |
| `ret` | `float*` | Output filtered image |

### Direct Kernel Access

```python
from yapocis.core import load_program
from yapocis.utils.basic import get_test_image
import numpy as np

# Load test image
img = get_test_image()
gray = np.mean(img, axis=2).astype(np.float32)

# Load the median3x3 interface
prog = load_program("median3x3")

# Call kernel directly (positional args)
filtered = prog.median3x3(gray)
print(f"Filtered shape: {filtered.shape}")
```

## See Also

- {doc}`gaussian` - Gaussian smoothing
- {doc}`anisodiff` - Edge-preserving diffusion
