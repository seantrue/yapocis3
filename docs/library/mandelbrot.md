# Mandelbrot

GPU-accelerated Mandelbrot set fractal computation.

## Overview

The `Mandelbrot` class provides GPU-accelerated computation of the Mandelbrot set, computing escape iterations for each point in the complex plane.

## Class Reference

```python
from yapocis.lib.mandelbrot import Mandelbrot

m = Mandelbrot()
```

## Usage

The Mandelbrot class wraps the `mandelbrot` kernel which takes a complex array and returns iteration counts.

```python
import numpy as np
from yapocis.lib.mandelbrot import Mandelbrot

# Create complex plane coordinates
width, height = 1024, 768
x = np.linspace(-2.5, 1.0, width)
y = np.linspace(-1.25, 1.25, height)
X, Y = np.meshgrid(x, y)
q = (X + 1j * Y).astype(np.complex64)

# Compute Mandelbrot set (positional args for kernel calls)
m = Mandelbrot()
iterations = m.kmgr.mandelbrot(q, 256)
print(f"Iterations shape: {iterations.shape}")
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `q` | complex64 array | Complex plane coordinates |
| `maxiter` | int | Maximum iterations before escape |

## Example: Rendering

```python {.no-exec}
import numpy as np
import matplotlib.pyplot as plt
from yapocis.lib.mandelbrot import Mandelbrot

def render_mandelbrot(center, zoom, width=1024, height=768, maxiter=256):
    # Compute bounds
    aspect = width / height
    half_width = 2.0 / zoom
    half_height = half_width / aspect

    x = np.linspace(center[0] - half_width, center[0] + half_width, width)
    y = np.linspace(center[1] - half_height, center[1] + half_height, height)
    X, Y = np.meshgrid(x, y)
    q = (X + 1j * Y).astype(np.complex64)

    # Compute on GPU
    m = Mandelbrot()
    iters = m.kmgr.mandelbrot(q, maxiter)  # Positional args

    # Normalize for display
    result = iters.astype(np.float32) / maxiter
    return result

# Full set view
img = render_mandelbrot(center=(-0.5, 0.0), zoom=1.0)
plt.imshow(img, cmap='hot')
plt.show()

# Zoomed into interesting region
img_zoomed = render_mandelbrot(
    center=(-0.7435, 0.1314),
    zoom=100,
    maxiter=500
)
```

## Algorithm

The Mandelbrot set is defined as points `c` in the complex plane where the iteration:

```
z_{n+1} = z_n^2 + c
```

Starting from `z_0 = 0` does not escape to infinity.

The kernel returns the number of iterations before `|z| > 2` (escape condition), or `maxiter` if the point doesn't escape.

## Coloring

The iteration count can be used for coloring:

```python {.no-exec}
# Linear normalization
normalized = iters / maxiter

# Log scaling for smoother bands
log_iters = np.log(iters + 1) / np.log(maxiter + 1)

# Smooth coloring using escape radius
# (requires modification to kernel to return final |z|)
```

## Performance

GPU computation provides significant speedup:

| Resolution | CPU Time | GPU Time | Speedup |
|------------|----------|----------|---------|
| 1024x768 | ~2.0s | ~0.05s | 40x |
| 4096x3072 | ~30s | ~0.3s | 100x |

*Times are approximate and depend on hardware and maxiter.*

## Underlying Kernel: mandelbrot

The `Mandelbrot` class uses the `mandelbrot` interface.

### Interface Definition

```
interface mandelbrot {
    kernel mandelbrot(in complex64 q, out_like short q, int maxiter);
};
```

### Kernel Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `q` | `complex64*` | Input complex coordinates |
| `output` | `short*` | Output iteration counts |
| `maxiter` | `int` | Maximum iterations before escape |

### Direct Kernel Access

```python
from yapocis.core import load_program
import numpy as np

# Load the mandelbrot interface
prog = load_program("mandelbrot")

# Create complex plane
x = np.linspace(-2.5, 1.0, 1024)
y = np.linspace(-1.25, 1.25, 768)
X, Y = np.meshgrid(x, y)
q = (X + 1j * Y).astype(np.complex64)

# Call kernel directly (positional args)
iterations = prog.mandelbrot(q, 256)  # q, maxiter
```

## See Also

- {doc}`simplex` - Procedural noise generation
