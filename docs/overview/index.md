# Overview

Yapocis is a multi-backend GPU/CPU computing framework for image processing and graphics operations. It provides a unified Python interface to OpenCL, CUDA, Metal, and Numba backends through an IDL-like interface definition language and Mako template system.

## What Problem Does Yapocis Solve?

Images are inherently computationally challenging. A single 4K image contains over 8 million pixels, and processing them efficiently requires careful attention to memory management, parallelization, and hardware-specific optimizations. Traditional approaches force developers to choose between:

- **Pure Python/NumPy**: Easy to write, but often too slow for real-time or batch processing
- **C/C++ extensions**: Fast, but painful to develop and maintain
- **GPU frameworks**: Blazingly fast, but each vendor (OpenCL, CUDA, Metal) requires different code

Yapocis addresses this by letting you write compute kernels once and execute them across different hardware platforms without code modification. Backend selection occurs at runtime, proving that Write Once, Run Anywhere can actually work if you control the right variables.

## Key Features

### IDL-Based Interface System

Define kernel interfaces once using an intuitive interface definition language:

```
interface mandelbrot {
    kernel mandelbrot(in complex64 q, out_like short q, in int32 maxiter);
};
```

The framework automatically handles:

- **Memory allocation and management** - the hard parts
- **Data transfer between host and device** - the slow parts
- **Type marshaling and buffer hints** - the error-prone parts
- **Multi-dimensional array handling** - the confusing parts

### Multi-Backend Architecture

Execute the same code across different compute platforms:

```python {.no-exec}
from yapocis.core import manager

# Select backend programmatically
prog = manager.load_program("mandelbrot", backend="metal")
result = prog.mandelbrot(complex_grid, 1000)  # positional args

# Or use environment variable: export YAPOCIS_BACKEND=ocl
```

**Supported Backends:**

| Backend | Via | Best For |
|---------|-----|----------|
| OpenCL | PyOpenCL | Cross-platform GPU/CPU support |
| Metal | PyMetallic | Apple GPU acceleration |
| CUDA | PyCUDA | NVIDIA GPU support |
| Numba | Numba JIT | CPU fallback with JIT compilation |

### High-Level Libraries

Pre-built implementations for common operations:

- **Graphics**: Lines, circles, rectangles, text rendering with font and rotation support
- **Image Processing**: Convolution, Gaussian blur, gradients, median filtering
- **Color**: RGB/HSI conversions, color math transformations
- **Edge Detection**: Zero crossing, anisotropic diffusion
- **Fractals**: Mandelbrot set computation
- **Path Tracing**: Signed distance field rendering with Jump Flood Algorithm

### Performance Tracking

Built-in performance infrastructure with DuckDB storage for:

- Cross-backend performance comparison
- Longitudinal trend analysis
- Automated regression detection

## Who Is Yapocis For?

Yapocis is designed for:

- **Researchers** exploring image processing algorithms who need fast iteration
- **Developers** building applications that must run across different GPU hardware
- **Hobbyists** experimenting with GPU computing without vendor lock-in
- **Teams** who want to test on CPU (Numba) and deploy on GPU (OpenCL/CUDA/Metal)

## Quick Example

```python
import numpy as np
from yapocis.core import manager
from yapocis.lib.gaussian import Gaussian
from yapocis.utils.basic import get_test_image

# Load a kernel
prog = manager.load_program("mandelbrot")

# Generate Mandelbrot set
x, y = np.mgrid[-2:2:512j, -2:2:512j]
q = (x + 1j * y).astype(np.complex64)
result = prog.mandelbrot(q, 256)  # positional args
print(f"Mandelbrot result: {result.shape}, max iterations: {result.max()}")

# Or use the high-level library
img = get_test_image()
gaussian = Gaussian()
blurred = gaussian.gauss_image(img[:,:,0], scale=3)
print(f"Blurred shape: {blurred.shape}")
```

## Next Steps

- {doc}`/quickstart/installation` - Install yapocis and backend dependencies
- {doc}`/quickstart/first-program` - Run your first yapocis program
- {doc}`/library/index` - Explore the built-in library modules
- {doc}`/kernel-quickstart/index` - Learn to write your own kernels
