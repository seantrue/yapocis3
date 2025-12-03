# Image Operations

GPU-accelerated image operations including histogram computation.

## Overview

The `Imageops` class provides GPU-accelerated image operations, currently focused on histogram computation.

## Class Reference

```python
from yapocis.lib.imageops import Imageops

ops = Imageops()
```

## Methods

### histogram

Compute histogram of an image.

```python
hist = ops.histogram(image, n_bins=256, low=0.0, high=1.0)
```

**Parameters:**
- `input` - 2D float32 array
- `n_bins` - Number of histogram bins (default: 256)
- `n_workers` - GPU worker count (default: 32)
- `low` - Minimum value for binning (default: 0.0)
- `high` - Maximum value for binning (default: 1.0)

**Returns:** int32 array of bin counts

### histogram_res

Resident mode version for repeated histogram operations.

```python
hist = np.zeros(256, dtype=np.int32)
ops.histogram_res(image, hist, low=0.0, high=1.0)
```

## Usage

```python
import numpy as np
from yapocis.lib.imageops import Imageops
from yapocis.utils.basic import get_test_image

# Load image and convert to grayscale
img = get_test_image()
gray = np.mean(img, axis=2).astype(np.float32)

# Compute histogram
ops = Imageops()
hist = ops.histogram(gray, n_bins=256)

# Find most common intensity
peak_bin = np.argmax(hist)
peak_intensity = peak_bin / 256.0
```

## Example: Histogram Equalization

```python
import numpy as np
from yapocis.lib.imageops import Imageops
from yapocis.lib.channels import histeq
from yapocis.utils.basic import get_test_image

# Load image and convert to grayscale (simulate low contrast)
img = get_test_image()
gray = np.mean(img, axis=2).astype(np.float32) * 0.5  # Low contrast

# Compute histogram before
ops = Imageops()
hist_before = ops.histogram(gray)

# Equalize
equalized = histeq(gray)

# Compute histogram after
hist_after = ops.histogram(equalized)
```

## Example: Automatic Threshold (Otsu)

```python
import numpy as np
from yapocis.lib.imageops import Imageops

def otsu_threshold(image):
    """Find optimal threshold using Otsu's method."""
    ops = Imageops()
    hist = ops.histogram(image, n_bins=256)

    # Normalize histogram
    pdf = hist.astype(np.float64) / hist.sum()

    # Compute Otsu threshold
    bin_centers = np.linspace(0, 1, 256)

    best_threshold = 0
    best_variance = 0

    for t in range(1, 255):
        w0 = pdf[:t].sum()
        w1 = pdf[t:].sum()

        if w0 == 0 or w1 == 0:
            continue

        mu0 = (pdf[:t] * bin_centers[:t]).sum() / w0
        mu1 = (pdf[t:] * bin_centers[t:]).sum() / w1

        variance = w0 * w1 * (mu0 - mu1) ** 2

        if variance > best_variance:
            best_variance = variance
            best_threshold = bin_centers[t]

    return best_threshold

# Usage
from yapocis.utils.basic import get_test_image

img = get_test_image()
gray = np.mean(img, axis=2).astype(np.float32)
threshold = otsu_threshold(gray)
binary = (gray > threshold).astype(np.float32)
```

## Performance Notes

- GPU histogram uses atomic operations for bin counting
- Best performance with power-of-2 bin counts
- Resident mode avoids repeated buffer allocation

## Underlying Kernel: imageops

The `Imageops` class uses the `imageops` interface.

### Interface Definition

```
interface imageops {
    kernel clamp_pixels(in float input, float low, float high, out_like input);
    kernel stretch(in float input, float input_min, float input_max,
                   float low, float high, out_like input);
    kernel binarize(in float input, float threshold, out_like input);
    kernel histogram(in float input, size_of input, float low, float high,
                     out int hst, size_of hst);
    alias histogram_res as histogram(resident float input, size_of input,
                     float low, float high, resident int hst, size_of hst);
};
```

### Kernel Parameters

**histogram:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `float*` | Input image |
| `input_size` | `int` | Total pixels (derived from `size_of input`) |
| `low` | `float` | Minimum value for binning |
| `high` | `float` | Maximum value for binning |
| `hst` | `int*` | Output histogram bins |
| `hst_size` | `int` | Number of bins (derived from `size_of hst`) |

**clamp_pixels:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `float*` | Input image |
| `low` | `float` | Minimum clamp value |
| `high` | `float` | Maximum clamp value |

**binarize:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `float*` | Input image |
| `threshold` | `float` | Binarization threshold |

### Direct Kernel Access

```python
from yapocis.core import load_program
import numpy as np

# Load the imageops interface
prog = load_program("imageops")

# Histogram (positional args)
hist = np.zeros(256, dtype=np.int32)
prog.histogram(image, 0.0, 1.0, hist)

# Clamp pixels
clamped = prog.clamp_pixels(image, 0.0, 1.0)

# Binarize
binary = prog.binarize(image, 0.5)

# Stretch contrast
stretched = prog.stretch(image, 0.2, 0.8, 0.0, 1.0)
```

## See Also

- {doc}`channels` - Channel utilities including histeq
- {doc}`gradient` - Image gradients
