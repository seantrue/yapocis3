# Library Reference

High-level Python APIs for image processing and GPU computing.

## Overview

The `yapocis.lib` module provides high-level wrappers around GPU kernels. These classes handle:
- Buffer management
- Parameter validation
- Type coercion
- Backend abstraction

All library classes inherit from `BoundKernels`, which provides automatic kernel loading and buffer management.

## Usage Pattern

```python
from yapocis.lib.gradient import Gradient

# Create instance (loads kernels)
grad = Gradient()

# Call methods on your data
magnitude, direction = grad.gradient(image, stride=3)
```

## Module Categories

### Image Filtering
- {doc}`anisodiff` - Perona-Malik anisotropic diffusion
- {doc}`gaussian` - Gaussian blur and DoG edge detection
- {doc}`median` - Median filtering for noise removal
- {doc}`convolve` - General 1D/2D convolution
- {doc}`convolve-kernels` - Predefined convolution kernels

### Color Processing
- {doc}`colormath` - Color space transformations (RGB, XYZ, Lab)
- {doc}`hsi` - HSI color space conversion
- {doc}`channels` - Channel manipulation utilities

### Edge & Feature Detection
- {doc}`gradient` - Image gradient computation
- {doc}`zc` - Zero-crossing detection

### Graphics & Drawing
- {doc}`graphics` - 2D primitives (1/3 channel images)
- {doc}`kerneldraw` - Low-level drawing kernels
- {doc}`typeset` - Text rendering

### Math & Computation
- {doc}`operators` - Element-wise operations
- {doc}`mandelbrot` - Fractal generation
- {doc}`simplex` - Simplex noise generation
- {doc}`fft` - Fourier transforms

### Image Operations
- {doc}`imageops` - Resize, rotate, composite

### Tracing & Vectorization
- {doc}`truetrace` - Bitmap to vector contour tracing

### Deep Learning
- {doc}`vgg16` - VGG16 feature extraction
- {doc}`resnet50` - ResNet50 inference

### Utilities
- {doc}`mplbackend` - Matplotlib integration

```{toctree}
:maxdepth: 1
:hidden:

anisodiff
channels
colormath
convolve
convolve-kernels
fft
gaussian
gradient
graphics
hsi
imageops
kerneldraw
mandelbrot
median
mplbackend
operators
resnet50
simplex
truetrace
typeset
vgg16
zc
```
