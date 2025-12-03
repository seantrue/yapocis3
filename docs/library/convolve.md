# Convolution

1D convolution operations with named kernel support.

## Overview

The `Convolve1D` class provides GPU-accelerated 1D convolution using named kernels. It works with the `convolve` interface which supports pre-defined convolution kernels.

## Class Reference

```python {.no-exec}
from yapocis.lib.convolve import Convolve1D

conv = Convolve1D(name="box3", conv=kernel_array)
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Required. Name identifier for the kernel |
| `conv` | ndarray | Required. 1D float32 kernel weights |

## Usage

```python
import numpy as np
from yapocis.lib.convolve import Convolve1D
from yapocis.lib.convolve_kernels import generate_kernel

# Generate a Hamming window kernel
kernel = generate_kernel("hamming", 5)
print(f"Hamming kernel: {kernel}")

# Create the convolution wrapper
conv = Convolve1D(name="hamming5", conv=kernel)

# Apply convolution to signal
signal = np.random.rand(1024).astype(np.float32)
result = conv.kmgr.hamming5(signal)  # Use the named kernel method
print(f"Result shape: {result.shape}")
```

## Integration with Kernel Generators

The `Convolve1D` class is typically used with kernels from `convolve_kernels.py`:

```python
import numpy as np
from yapocis.lib.convolve import Convolve1D
from yapocis.lib.convolve_kernels import (
    generate_hamming, generate_hann, generate_box
)

# Smoothing with Hamming window
hamming = generate_hamming(9)
conv = Convolve1D(name="hamming9", conv=hamming.astype(np.float32))
print(f"Hamming9 kernel created, size: {len(hamming)}")

# Box filter (uniform)
box = generate_box(5)
conv = Convolve1D(name="box5", conv=box.astype(np.float32))
print(f"Box5 kernel created, size: {len(box)}")
```

## Underlying Kernel: convolve

The `Convolve1D` class uses the `convolve` interface.

### Interface Definition

```
interface convolve {
    kernel ${name}(size_of a, in float a, out_like a);
    kernel ${name}_fast(size_of a, in float a, out_like a, int chunk);
};
```

This is a templated interface—`${name}` is replaced with the kernel name at load time.

### Kernel Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `size` | `int` | Signal length (derived from `size_of a`) |
| `a` | `float*` | Input signal |
| `output` | `float*` | Convolved output |
| `chunk` | `int` | Chunk size for fast variant |

### Direct Kernel Access

```python
from yapocis.core import load_program
import numpy as np

# Load with specific kernel configuration
kernel_weights = np.array([0.25, 0.5, 0.25], dtype=np.float32)
prog = load_program("convolve", name="smooth3", conv=kernel_weights)

# Call kernel directly (positional args)
signal = np.random.rand(1024).astype(np.float32)
result = prog.smooth3(signal)
print(f"Convolved signal shape: {result.shape}")
```

### Named Kernel Loading

Each convolution kernel must be loaded with its weights:

```python
from yapocis.core import load_program
from yapocis.lib.convolve_kernels import generate_hamming
import numpy as np

# Generate kernel weights
weights = generate_hamming(9).astype(np.float32)

# Load with specific name
prog = load_program("convolve", name="hamming9", conv=weights)

# Apply to signal
signal = np.random.rand(1024).astype(np.float32)
convolved = prog.hamming9(signal)
print(f"Convolved shape: {convolved.shape}")
```

## See Also

- {doc}`convolve-kernels` - Kernel generation utilities
- {doc}`gaussian` - 2D Gaussian convolution
