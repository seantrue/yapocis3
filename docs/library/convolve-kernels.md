# Convolution Kernels

1D kernel generation utilities for signal processing.

## Overview

The `convolve_kernels` module provides functions for generating various 1D convolution kernels including windows, smoothing filters, and derivative operators.

## Available Kernels

### Window Functions

| Function | Description |
|----------|-------------|
| `generate_hamming(size)` | Hamming window |
| `generate_hann(size)` | Hann (Hanning) window |
| `generate_blackman(size)` | Blackman window |
| `generate_bartlett(size)` | Bartlett (triangular) window |

### Smoothing Filters

| Function | Description |
|----------|-------------|
| `generate_box(size)` | Uniform (box) filter |
| `generate_triangle(size)` | Triangular filter |
| `generate_binomial(size)` | Binomial coefficients |

### Derivative Operators

| Function | Description |
|----------|-------------|
| `generate_deriv1(size)` | First derivative (central difference) |
| `generate_deriv2(size)` | Second derivative (Laplacian) |
| `generate_sobel(size)` | Sobel derivative |
| `generate_prewitt(size)` | Prewitt derivative |

### Sharpening Filters

| Function | Description |
|----------|-------------|
| `generate_sharpen(size)` | Sharpening kernel |
| `generate_unsharp(size)` | Unsharp masking |
| `generate_dog(size)` | Difference of Gaussians |

## Usage

```python
from yapocis.lib.convolve_kernels import (
    generate_hamming, generate_deriv1, generate_kernel
)

# Generate specific kernel
hamming_5 = generate_hamming(5)
deriv_3 = generate_deriv1(3)

# Use generic generator
kernel = generate_kernel("blackman", 9)
```

## Kernel Generator Function

```python
from yapocis.lib.convolve_kernels import generate_kernel

# Available types: hamming, hann, blackman, bartlett, box,
# triangle, binomial, deriv1, deriv2, sobel, prewitt,
# sharpen, unsharp, dog

kernel = generate_kernel("hamming", 5)  # Returns float32 array
```

## Example: Signal Smoothing

```python
import numpy as np
from yapocis.lib.convolve_kernels import generate_hamming

# Generate noisy signal
t = np.linspace(0, 1, 1000)
signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(1000)

# Create Hamming window kernel
kernel = generate_hamming(15)

# Convolve (using numpy for illustration)
smoothed = np.convolve(signal, kernel, mode='same')
```

## Example: Edge Detection

```python
from yapocis.lib.convolve_kernels import generate_sobel

# Get Sobel derivative kernel
sobel = generate_sobel(3)  # [-1, 0, 1]

# Apply to signal
edges = np.convolve(signal, sobel, mode='same')
```

## YAML Specification

Kernels can be specified via YAML for batch generation:

```python
from yapocis.lib.convolve_kernels import generate_convs_from_spec

spec = {
    "windows": [{"type": "hamming", "sizes": [5, 9, 13]}],
    "smoothing": [{"type": "box", "sizes": [3, 5]}],
    "derivatives": [{"type": "deriv1", "sizes": [3, 5]}]
}

convs = generate_convs_from_spec(spec)
# Returns: [("hamming5", array), ("hamming9", array), ...]
```

## See Also

- {doc}`convolve` - 1D convolution class
- {doc}`gaussian` - 2D Gaussian convolution
