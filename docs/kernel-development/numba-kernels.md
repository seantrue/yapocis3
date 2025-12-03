# Numba Kernels

Numba kernels are pure Python implementations using NumPy and optional JIT compilation.

## Overview

Unlike OpenCL/CUDA/Metal which use Mako templates, Numba kernels are regular Python modules:

```
yapocis/kernels/numbax/
├── __init__.py
├── gradient.py
├── mandelbrot.py
└── ...
```

## Basic Structure

```python
from __future__ import annotations
import numpy as np

def sum(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    np.add(a, b, out=c)
    return c
```

Key points:
- Function names must match the kernel names in the interface
- Parameters follow the interface specification order
- Return the output array(s)

## JIT Compilation

Use Numba's `@njit` decorator for performance:

```python {.no-exec}
from __future__ import annotations
import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def _wrap(f):
            return f
        return _wrap


@njit(cache=True, fastmath=True)
def _gradient_njit(img: np.ndarray, reach: int,
                   grad: np.ndarray, theta: np.ndarray) -> None:
    h, w = img.shape
    for y in range(h):
        for x in range(w):
            # ... pixel-by-pixel computation ...
            pass
```

The fallback decorator ensures code works without Numba installed.

## Hybrid Implementation Pattern

Provide both JIT and pure NumPy paths:

```python
def gradient(width: int, height: int, input: np.ndarray,
             reach: int, grad: np.ndarray, theta: np.ndarray):
    if HAVE_NUMBA:
        _gradient_njit(input, int(reach), grad, theta)
    else:
        # Vectorized NumPy fallback
        gx = input[:, reach:] - input[:, :-reach]
        gy = input[reach:, :] - input[:-reach, :]
        # ... vectorized computation ...
    return grad, theta
```

## Extension Variants

Interface extensions (like `_res` for resident buffers) need matching functions:

```python
def gradient(width, height, input, reach, grad, theta):
    # Main implementation
    ...
    return grad, theta


def gradient_res(width, height, input, reach, grad, theta):
    # Resident variant - often just calls main
    return gradient(width, height, input, reach, grad, theta)
```

## Row-Major Order

Numba uses NumPy's row-major (C) order, matching CUDA:

```python {.no-exec}
# NumPy indexing: array[row, col] = array[y, x]
for y in range(height):
    for x in range(width):
        pixel = img[y, x]
```

This differs from OpenCL/Metal column-major order, but the framework handles the conversion.

## Type Annotations

Use proper type hints for clarity:

```python
def process(
    width: int,
    height: int,
    input: np.ndarray,
    output: np.ndarray,
) -> np.ndarray:
    ...
```

## Complete Example

```python {.no-exec}
from __future__ import annotations
import math
import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def _wrap(f):
            return f
        return _wrap


@njit(cache=True, fastmath=True)
def _clamp(v: int, lo: int, hi: int) -> int:
    if v < lo:
        return lo
    elif v > hi:
        return hi
    else:
        return v


@njit(cache=True, fastmath=True)
def _gradient_njit(
    img: np.ndarray, reach: int, grad: np.ndarray, theta: np.ndarray
) -> None:
    h, w = img.shape
    for y in range(h):
        for x in range(w):
            if (x - reach) >= 0 and (x + reach) < w and \
               (y - reach) >= 0 and (y + reach) < h:
                gx = img[y, x + reach] - img[y, x - reach]
                gy = img[y + reach, x] - img[y - reach, x]
                g = math.sqrt(gx * gx + gy * gy)
                th = (-math.atan2(gy, gx) / math.pi) + 0.5
                grad[y, x] = g if not math.isnan(g) else 0.0
                theta[y, x] = th if not math.isnan(th) else 0.0
            else:
                grad[y, x] = 0.0
                theta[y, x] = 0.0


def gradient(
    width: int,
    height: int,
    input: np.ndarray,
    reach: int,
    grad: np.ndarray,
    theta: np.ndarray,
):
    if HAVE_NUMBA:
        _gradient_njit(input.astype(np.float32, copy=False),
                       int(reach), grad, theta)
    else:
        # Vectorized NumPy fallback
        img = input
        h, w = img.shape
        grad[:, :] = 0.0
        theta[:, :] = 0.0
        r = int(reach)
        if r > 0:
            gx = img[:, r:] - img[:, :-r]
            gy = img[r:, :] - img[:-r, :]
            g = np.zeros_like(img)
            g[r:-r, r:-r] = np.sqrt(gx[r:-r, :-r] ** 2 +
                                    gy[:-r, r:-r] ** 2)
            grad[:, :] = g
            # ... theta computation ...
    return grad, theta


def gradient_res(width, height, input, reach, grad, theta):
    return gradient(width, height, input, reach, grad, theta)
```

## Multi-Channel Support

For multi-channel images, handle the extra dimension:

```python
def gradient4(width, height, input, reach, grad, theta):
    # input shape: (height, width, 4)
    for c in range(4):
        grad[:, :, c], theta[:, :, c] = _process_channel(
            input[:, :, c], reach
        )
    return grad, theta
```

## Performance Tips

1. **Use `@njit(cache=True)`** - Cache compiled functions
2. **Use `fastmath=True`** - Allow fast math optimizations
3. **Prefer explicit loops** - Numba optimizes loops well
4. **Avoid object mode** - Ensure all operations are JIT-compatible
5. **Pre-allocate outputs** - Avoid allocations in hot paths

## Testing Numba Kernels

```python
import pytest
from yapocis.core import manager

@pytest.mark.numba
def test_gradient_numba():
    prog = manager.load_program("gradient", backend="numba")
    img = np.random.rand(256, 256).astype(np.float32)
    grad, theta = prog.gradient(img, reach=1)
    assert grad.shape == img.shape
    assert not np.any(np.isnan(grad))
```

## Debugging

Pure Python makes debugging straightforward:

```python {.no-exec}
@njit(cache=True)
def _process_njit(data, output):
    for i in range(len(data)):
        # Temporarily remove @njit to use print()
        output[i] = data[i] * 2.0

# Or use Numba's debug mode
from numba import config
config.DEBUGINFO = 1
```
