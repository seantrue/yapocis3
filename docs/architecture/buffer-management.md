# Buffer Management

Device buffers, lifecycle, and read/write/ensure operations.

## Overview

Yapocis manages GPU buffer allocation, transfer, and lifecycle automatically. The buffer system uses a `WeakValueDictionary` to track buffers by their NumPy array identity, enabling automatic reuse without manual memory management.

## Buffer Lifecycle

```
          write()                kernel()              read()
   NumPy ──────────► GPU Buffer ──────────► GPU Buffer ──────────► NumPy
   Array            (allocated)             (modified)             Array
```

### Key Operations

| Operation | Description |
|-----------|-------------|
| `write(array)` | Transfer NumPy array to GPU buffer |
| `read(array)` | Transfer GPU buffer back to NumPy array |
| `ensure(array)` | Create buffer if not exists (no transfer) |
| `purge()` | Release all GPU buffers |

## Buffer Tracking

Buffers are tracked by the `id()` of their corresponding NumPy array:

```python {.no-exec}
# Internally, buffers are stored like this:
self._buffers = weakref.WeakValueDictionary()

def write(self, array):
    key = id(array)
    if key not in self._buffers:
        self._buffers[key] = self._create_buffer(array)
    # Transfer data to buffer
```

This enables:
- **Automatic reuse**: Same array → same buffer
- **Automatic cleanup**: When NumPy array is garbage collected, buffer is released
- **No manual tracking**: Users just pass arrays

## Resident Mode

For repeated operations on the same data, use **resident mode** to avoid redundant transfers:

```python {.no-exec}
from yapocis.lib.gaussian import Gaussian

g = Gaussian()

# Upload once
g.write(image)

# Multiple operations without re-upload
for i in range(10):
    g.kmgr.gauss_sep_res(image, output, conv, scale=2+i)

# Download once
result = g.read(output)
```

### Resident Kernel Variants

Most kernels have a `_res` suffix variant for resident operation:

| Standard | Resident | Description |
|----------|----------|-------------|
| `gauss_sep` | `gauss_sep_res` | Separable Gaussian |
| `lines` | `lines_res` | Line drawing |
| `add` | `add_res` | Array addition |

## Async Writes

For overlapping computation with data transfer:

```python {.no-exec}
# Enable async writes globally
import os
os.environ['YAPOCIS_ASYNC_WRITE'] = '1'

# Or per-instance
g = Gaussian(async_write=True)

# Write returns immediately (non-blocking)
g.write(large_array)

# ... computation can proceed while transfer happens ...

# Kernel execution will wait for transfer if needed
g.gauss_image(large_array, scale=5)
```

## Memory Considerations

### GPU Memory Limits

Monitor GPU memory with backend-specific tools:

```python {.no-exec}
# OpenCL - check device memory
import pyopencl as cl
context = cl.create_some_context()
device = context.devices[0]
print(f"Global memory: {device.global_mem_size / 1e9:.1f} GB")

# Metal - memory is unified on Apple Silicon
# CUDA
import pycuda.autoinit
device = pycuda.autoinit.device
print(f"Total memory: {device.total_memory() / 1e9:.1f} GB")
```

### Best Practices

1. **Reuse buffers**: Pass the same array to avoid reallocation
2. **Use resident mode**: For iterative algorithms
3. **Purge when done**: Call `purge()` or use context managers
4. **Size appropriately**: Large arrays consume GPU memory

## Context Manager

Use context managers for automatic cleanup:

```python {.no-exec}
from yapocis.lib.graphics4 import Graphics4

with Graphics4() as g:
    g.circles(canvas, circles, color=red)
    result = g.read(canvas)
# Buffers automatically released here
```

## Buffer Data Types

Yapocis primarily uses:

| NumPy dtype | GPU Type | Notes |
|-------------|----------|-------|
| `float32` | `float` | Standard for images |
| `complex64` | `float2` | FFT operations |
| `int32` | `int` | Indices, histograms |
| `uint8` | `uchar` | Raw image data (converted internally) |

Arrays are automatically made contiguous:

```python {.no-exec}
# This works even for non-contiguous arrays
sliced = image[::2, ::2]  # Non-contiguous view
result = g.gauss_image(sliced)  # Automatically copied to contiguous
```

## Debugging Buffers

Set environment variable for buffer tracking:

```bash
export YAPOCIS_DEBUG_BUFFERS=1
```

This logs buffer operations:
```
[BUFFER] write: array id=140234567890 shape=(512, 512) dtype=float32
[BUFFER] reuse: array id=140234567890 (existing buffer)
[BUFFER] read: array id=140234567890
```
