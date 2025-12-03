# Bound Kernels

BoundKernel pattern and usage.

## Overview

`BoundKernels` is the base class for all high-level library modules. It binds an interface to a backend KernelManager, providing a clean API for users while handling backend selection, device management, and buffer operations automatically.

## Class Definition

```python {.no-exec}
class BoundKernels:
    INTERFACE = None  # Subclasses set this
    load_program_config = {}
    use_async = None
    use_async_write = None

    def __init__(
        self,
        device_name: str | int | None = None,
        backend: str | None = None,
        use_async: bool | None = None,
        async_write: bool | None = None,
        **kwargs
    ):
        self.kmgr = load_program(
            self.INTERFACE,
            backend=backend,
            device_name=device_name,
            async_write=self.async_write,
            **self.load_program_config,
            **kwargs
        )
```

## Creating a Library Module

```python {.no-exec}
from yapocis.lib.bound_kernels import BoundKernels

class MyProcessor(BoundKernels):
    INTERFACE = "myprocessor"  # Uses myprocessor.interface

    def process(self, image, parameter):
        """High-level method for users."""
        # Call kernel via kernel manager
        return self.kmgr.process_kernel(image, parameter)

    def process_batch(self, images):
        """Batch processing with resident buffers."""
        results = []
        for img in images:
            self.write(img)
            self.kmgr.process_kernel_res(img)
            results.append(self.read(img))
        return results
```

## Attribute Delegation

Unrecognized attributes are delegated to the kernel manager:

```python {.no-exec}
class Gaussian(BoundKernels):
    INTERFACE = "convolvesep"

# These are equivalent:
g = Gaussian()
g.gauss_sep(image, conv, scale=3)  # Auto-delegated
g.kmgr.gauss_sep(image, conv, scale=3)  # Explicit
```

## Configuration Options

### Backend Selection

```python {.no-exec}
# Use environment variable
import os
os.environ['YAPOCIS_BACKEND'] = 'metal'
g = Gaussian()  # Uses Metal

# Explicit backend
g = Gaussian(backend='cuda')  # Uses CUDA

# Device selection
g = Gaussian(device_name='Apple M1 Pro')
g = Gaussian(device_name=0)  # First device
```

### Async Configuration

Three-level resolution (highest to lowest priority):

1. **Instance parameter**
2. **Class variable**
3. **Environment variable**

```python {.no-exec}
# Environment level
os.environ['YAPOCIS_ASYNC'] = '1'

# Class level
class FastGaussian(Gaussian):
    use_async = True

# Instance level
g = Gaussian(use_async=True)
```

### Load-time Configuration

```python {.no-exec}
class FFT(BoundKernels):
    INTERFACE = "fft"

    # Passed to load_program
    load_program_config = {
        "options": ["-cl-opt-disable"]  # OpenCL compiler options
    }
```

## Buffer Operations

```python {.no-exec}
class MyModule(BoundKernels):
    INTERFACE = "mymodule"

    def process(self, image):
        # Write to GPU
        self.write(image)

        # Call resident kernel
        self.kmgr.kernel_res(image)

        # Read back
        return self.read(image)

    def cleanup(self):
        # Release all GPU buffers
        self.purge()
```

## Context Manager

```python {.no-exec}
with Gaussian() as g:
    result = g.gauss_image(image, scale=5)
# Automatic purge on exit
```

Equivalent to:
```python {.no-exec}
g = Gaussian()
try:
    result = g.gauss_image(image, scale=5)
finally:
    g.purge()
```

## Cleanup Behavior

By default, buffers are NOT automatically purged when the instance is deleted. Enable with:

```bash
export YAPOCIS_AUTO_PURGE=1
```

Or use explicit cleanup:
```python {.no-exec}
g = Gaussian()
# ... use g ...
g.purge()  # Explicit cleanup
del g
```

## Example: Complete Library Module

```python {.no-exec}
from yapocis.lib.bound_kernels import BoundKernels
import numpy as np

class EdgeDetector(BoundKernels):
    """Edge detection using Sobel operators."""

    INTERFACE = "gradient"

    def detect_edges(self, image, threshold=0.1):
        """
        Detect edges in an image.

        Args:
            image: Grayscale float32 image
            threshold: Edge strength threshold

        Returns:
            Binary edge image
        """
        # Ensure correct dtype
        image = np.ascontiguousarray(image, dtype=np.float32)

        # Allocate output
        gradient = np.zeros_like(image)
        direction = np.zeros_like(image)

        # Call kernel
        self.kmgr.gradient(image, gradient, direction)

        # Threshold
        edges = (gradient > threshold).astype(np.float32)
        return edges

    def detect_edges_resident(self, image, output, threshold=0.1):
        """Resident version for repeated calls."""
        gradient = np.zeros_like(image)
        direction = np.zeros_like(image)

        # Use resident kernels
        self.kmgr.gradient_res(image, gradient, direction)

        # Threshold on CPU (could also be a kernel)
        output[:] = (gradient > threshold).astype(np.float32)
        return output
```

## Best Practices

1. **Set INTERFACE**: Every subclass must set the `INTERFACE` class variable
2. **Use docstrings**: Document high-level methods for users
3. **Handle dtypes**: Ensure arrays are correct dtype before kernel calls
4. **Provide resident variants**: For performance-critical code paths
5. **Use context managers**: For automatic cleanup in scripts
