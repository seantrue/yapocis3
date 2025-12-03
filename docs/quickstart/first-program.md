# Your First Yapocis Program

This guide walks you through running your first yapocis program—generating a Mandelbrot set fractal.

## The Simplest Program

```python
import numpy as np
from yapocis.core import manager

# Create a complex grid for the Mandelbrot set
grid = np.mgrid[-2:1:512j, -1.5:1.5:512j]
q = (grid[0] + 1j * grid[1]).astype(np.complex64)

# Load the mandelbrot kernel
prog = manager.load_program("mandelbrot")

# Run it (positional arguments for kernel calls)
result = prog.mandelbrot(q, 256)

# result is now a 512x512 array of iteration counts
print(f"Result shape: {result.shape}")
print(f"Max iterations reached: {result.max()}")
```

## What Just Happened?

1. **Created input data**: A complex-valued grid spanning the interesting region of the Mandelbrot set
2. **Loaded a kernel**: `load_program("mandelbrot")` finds the interface definition and compiles the kernel
3. **Executed**: The kernel runs on your GPU (or CPU via Numba), computing iteration counts for each point
4. **Got results**: The output array is automatically transferred back to CPU memory

## Using High-Level Libraries

For common operations, yapocis provides high-level library classes:

```python
import numpy as np
from yapocis.lib.gaussian import Gaussian
from yapocis.utils.basic import get_test_image

# Load test image
img = get_test_image()

# Create Gaussian blur instance
gaussian = Gaussian()

# Apply Gaussian blur with scale=3 (kernel half-width)
# gauss_image operates on grayscale images
blurred = gaussian.gauss_image(img[:,:,0], scale=3)

print(f"Input shape: {img.shape}")
print(f"Output shape: {blurred.shape}")
```

## Drawing Graphics

```python
import numpy as np
from yapocis.lib.graphics4 import Graphics4

# Create a canvas (height, width, 4 channels for RGBA)
canvas = np.zeros((512, 512, 4), dtype=np.float32)

# Create graphics context
g = Graphics4()

# Draw circles
circles = [
    (256, 256, 100),  # center x, center y, radius
    (128, 128, 50),
    (384, 384, 75),
]
g.circles(canvas, circles, color=[1.0, 0.0, 0.0, 1.0])  # Red

# Draw lines
lines = [
    (0, 0, 511, 511),     # x1, y1, x2, y2
    (0, 511, 511, 0),
]
g.lines(canvas, lines, color=[0.0, 1.0, 0.0, 1.0])  # Green

# Read result back
result = g.read(canvas)
```

## Selecting a Backend

By default, yapocis picks the first available backend. You can specify one:

```python
# Via environment variable (before importing yapocis)
import os
os.environ["YAPOCIS_BACKEND"] = "metal"

# Or programmatically
from yapocis.core import manager
prog = manager.load_program("mandelbrot", backend="ocl")
```

## Saving Results

```python {.no-exec}
from PIL import Image
import numpy as np

# Normalize to 0-255 range (assumes result from previous block)
output = (result / result.max() * 255).astype(np.uint8)

# Save as image
Image.fromarray(output).save("mandelbrot.png")
```

## Next Steps

- {doc}`backends` - Learn about different backends and when to use each
- {doc}`images` - Work with real images
- {doc}`/library/index` - Explore available library modules
