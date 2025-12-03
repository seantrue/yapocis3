# Kernel Program Architecture

KernelBase and KernelManager architecture.

## Overview

The `KernelManager` is the central class that connects interface definitions to executable kernels. It handles:

- Loading and parsing interface files
- Rendering Mako templates
- Compiling kernel code
- Managing buffer transfers
- Dispatching kernel calls

## Loading a Program

```python
from yapocis.core import load_program

# Load with auto-detected backend
prog = load_program("demo")
print(f"Loaded program: {type(prog).__name__}")
```

Backend and device options:

```python {.no-exec}
# Load with specific backend
prog = load_program("graphics4", backend="metal")

# Load with device selection
prog = load_program("graphics4", device_name="Apple M1 Pro")
```

## load_program Flow

```
load_program("graphics4")
    │
    ▼
┌───────────────────────────────────────┐
│  1. Detect Backend                     │
│     - Check YAPOCIS_BACKEND env var   │
│     - Check available backends        │
│     - Select first available          │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│  2. Find Interface File               │
│     - Search YAPOCIS_PROJECT          │
│     - Search yapocis/interfaces/      │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│  3. Parse Interface                    │
│     - PyParsing grammar               │
│     - Generate InterfaceSpec IR       │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│  4. Find Template                      │
│     - Search yapocis/kernels/{backend}│
│     - e.g., graphics4.mako            │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│  5. Render Template                    │
│     - Mako template engine            │
│     - Pass interface context          │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│  6. Compile Kernels                    │
│     - Backend-specific compiler       │
│     - Cache compiled programs         │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│  7. Create KernelManager               │
│     - Wrap compiled program           │
│     - Set up callable interface       │
└───────────────────────────────────────┘
    │
    ▼
   Return KernelManager instance
```

## KernelManager Interface

The returned object provides:

```python {.no-exec}
prog = load_program("graphics4")

# Direct kernel calls
prog.lines(image, coords, widths, aa_widths, colors)

# Callable dictionary
prog.callable["lines"](image, coords, ...)

# Buffer operations
prog.write(array)
prog.read(array)
prog.ensure(array)
prog.purge()

# Async context
with prog.async_mode():
    prog.lines_res(image, ...)
    prog.circles_res(image, ...)
```

## Callable Dispatch

When you call a kernel, the KernelManager:

1. **Validates parameters** against interface spec
2. **Derives dimensions** (width_of, height_of)
3. **Ensures buffers** exist on GPU
4. **Writes input buffers** if needed
5. **Calculates global/local sizes**
6. **Dispatches kernel**
7. **Reads output buffers** if needed

```python {.no-exec}
# This call:
prog.lines(image, coords, widths, aa_widths, colors)

# Internally:
# 1. width = width_of(image) = image.shape[1]
# 2. height = height_of(image) = image.shape[0]
# 3. prog.write(coords)  # Input
# 4. prog.ensure(image)  # inout buffer
# 5. kernel.execute(global_size=(width*height,))
# 6. prog.read(image) if not resident
```

## Backend Managers

Each backend has its own manager implementation:

| Backend | Manager Class | Compiler |
|---------|---------------|----------|
| OpenCL | `OCLKernelManager` | PyOpenCL |
| CUDA | `CUDAKernelManager` | PyCUDA/NVCC |
| Metal | `MetalKernelManager` | pymetallic |
| Numba | `NumbaKernelManager` | Numba JIT |

## Caching

Compiled kernels are cached to avoid recompilation:

```bash
# Cache location (default)
~/.cache/yapocis/

# Custom cache
export YAPOCIS_CACHE=/path/to/cache
```

Cache keys include:
- Interface name
- Template hash
- Backend version
- Compiler options

## Global Size Calculation

For 2D image kernels:

```python {.no-exec}
# Standard (one thread per pixel)
global_size = (width * height,)

# Tiled (one thread per tile)
global_size = (ceil(width/tile_w) * ceil(height/tile_h),)

# 2D dispatch
global_size = (width, height)
```

The interface `meta` block can specify defaults:

```idl
meta {
    workgroup (32, 32, 1);
    tile_pixels (8, 8);
}
```

## Error Handling

```python
from yapocis.core import load_program

try:
    prog = load_program("nonexistent_interface_12345")
except ValueError as e:
    print("Interface not found")
```

```python {.no-exec}
try:
    prog.invalid_kernel(...)
except KeyError:
    print("Kernel not in interface")

try:
    prog.lines(wrong_args)
except ValueError:
    print("Parameter mismatch")
```

## Debugging

Enable debug mode for verbose output:

```python {.no-exec}
prog = load_program("graphics4", debug=True)
```

This prints:
- Template rendering
- Compiled source code
- Buffer operations
- Kernel dispatch info
