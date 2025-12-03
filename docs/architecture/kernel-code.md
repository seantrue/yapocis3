# Kernel Code Architecture

Backend-level kernel code (OpenCL/CUDA/Metal/Numba).

## Overview

Kernel code is stored as Mako templates (`.mako` files) for compiled backends (OpenCL, CUDA, Metal) or as Python modules (`.py` files) for the Numba backend. Templates are rendered at load time to produce backend-specific source code.

## Directory Structure

```
src/yapocis/kernels/
├── ocl/                    # OpenCL kernels
│   ├── constants.mako      # Common definitions
│   ├── helpers.mako        # Utility functions
│   ├── demo.mako
│   ├── graphics4.mako
│   └── ...
├── cuda/                   # CUDA kernels
│   ├── constants.mako      # CUDA-specific shims
│   ├── helpers.mako
│   ├── demo.mako
│   └── ...
├── metal/                  # Metal kernels
│   ├── constants.mako      # Metal-specific shims
│   ├── helpers.mako
│   ├── demo.mako
│   └── ...
└── numbax/                 # Numba kernels (Python)
    ├── demo.py
    ├── graphics4.py
    └── ...
```

## Template Resolution

When loading a program, yapocis searches for templates in this order:

1. `YAPOCIS_PROJECT` directory (if set)
2. Built-in kernel directory for the backend

```python
# Template search for "graphics4" on OpenCL:
# 1. $YAPOCIS_PROJECT/kernels/ocl/graphics4.mako
# 2. yapocis/kernels/ocl/graphics4.mako
```

## Template Structure

### Basic Template

```mako
<%inherit file="constants.mako"/>

## Kernel: demo
__kernel void demo(
    __global float *a,
    int width,
    int height
) {
    int xx = get_global_id(0);
    int yy = get_global_id(1);

    if (xx >= width || yy >= height) return;

    a[ADDRESS(xx, yy)] = 0.5f;
}
```

### With Mako Logic

```mako
<%inherit file="constants.mako"/>

## Parametric kernel generation
% for channels in [1, 2, 3, 4]:
__kernel void gradient${channels}(
    __global float *input,
    __global float *output,
    int width, int height
) {
    // ... kernel code for ${channels} channels

    % if channels > 1:
    // Multi-channel handling
    % endif
}
% endfor
```

## Backend Differences

### Address Calculation

OpenCL and Metal use **column-major** addressing:
```c
#define ADDRESS(xx, yy) ((xx) * height + (yy))
```

CUDA uses **row-major** addressing:
```c
#define ADDRESS(xx, yy) ((yy) * width + (xx))
```

### Work Item Access

| Backend | Get Global ID | Get Local ID |
|---------|---------------|--------------|
| OpenCL | `get_global_id(dim)` | `get_local_id(dim)` |
| CUDA | `blockIdx.x * blockDim.x + threadIdx.x` | `threadIdx.x` |
| Metal | `position.x` (from `[[thread_position_in_grid]]`) | `local_id.x` |

### Memory Qualifiers

| Concept | OpenCL | CUDA | Metal |
|---------|--------|------|-------|
| Global | `__global` | (default) | `device` |
| Local/Shared | `__local` | `__shared__` | `threadgroup` |
| Constant | `__constant` | `__constant__` | `constant` |
| Private | `__private` | (default) | `thread` |

## Shim Layer

The `constants.mako` file provides compatibility shims:

### OpenCL constants.mako
```mako
#define ADDRESS(xx, yy) ((xx) * height + (yy))

// Clamp macros
#define CLAMPX(x) clamp((x), 0, width - 1)
#define CLAMPY(y) clamp((y), 0, height - 1)
```

### CUDA constants.mako
```c
// Row-major addressing for CUDA
#define ADDRESS(xx, yy) ((yy) * width + (xx))

// Wrap OpenCL-style functions
#define get_global_id(d) ((d) == 0 ? blockIdx.x * blockDim.x + threadIdx.x : blockIdx.y * blockDim.y + threadIdx.y)
#define clamp(v, lo, hi) max((lo), min((hi), (v)))
```

### Metal constants.mako
```cpp
#define ADDRESS(xx, yy) ((xx) * height + (yy))

// Metal uses select() instead of ternary for vectors
#define TERNARY(cond, a, b) select((b), (a), (cond))
```

## Multi-Channel Kernels

Many kernels are generated for 1, 2, 3, and 4 channels:

```mako
% for N in [1, 2, 3, 4]:
__kernel void zcs${N}(
    __global float *input,
    __global float *output,
    int width, int height,
    float threshold
) {
    int xx = get_global_id(0);
    int yy = get_global_id(1);

    if (xx >= width || yy >= height) return;

    % for c in range(N):
    float v${c} = input[ADDRESS(xx, yy) * ${N} + ${c}];
    % endfor

    // Process channels...
}
% endfor
```

## Numba Kernels

Numba kernels are pure Python with optional JIT:

```python
from numba import njit
import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    def njit(func):
        return func

@njit
def demo_kernel(a):
    height, width = a.shape
    for yy in range(height):
        for xx in range(width):
            a[yy, xx] = 0.5

class DemoKernels:
    def demo(self, a):
        demo_kernel(a)
        return a
```

## Context Variables

Templates receive context from the interface and loader:

| Variable | Description |
|----------|-------------|
| `kernels` | List of kernel specs |
| `meta` | Interface metadata |
| `convs` | Convolution kernels (for convolve interfaces) |
| `debug` | Debug mode flag |

Access in templates:
```mako
% if meta.get('fast_math', False):
// Fast math enabled
% endif

% for kernel in kernels:
// Generating: ${kernel.name}
% endfor
```
