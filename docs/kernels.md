# Kernel Interfaces

Yapocis provides a collection of pre-built kernel interfaces for common image processing and graphics operations. Each interface is defined using the Yapocis IDL (Interface Definition Language) which specifies parameter types, data flow directions, and buffer management.

## Available Kernels

### Graphics Operations

- [graphics](kernels/graphics.md) - 2D graphics primitives (lines, circles, rectangles, triangles, ellipses)
- [graphics4](kernels/graphics4.md) - RGBA graphics primitives with 4-channel support
- [truetrace](kernels/truetrace.md) - Signed distance field path tracing

### Image Processing

- [imageops](kernels/imageops.md) - Basic image operations (clip, stretch, binarize, histogram)
- [median3x3](kernels/median3x3.md) - 3x3 median filter
- [gradient](kernels/gradient.md) - Image gradient computation
- [anisodiff](kernels/anisodiff.md) - Anisotropic diffusion filtering
- [zcs](kernels/zcs.md) - Zero-crossing detection and subtraction

### Convolution

- [convolvesep](kernels/convolvesep.md) - Separable convolution operations
- [convolves](kernels/convolves.md) - Multiple convolution kernels
- [convolve](kernels/convolve.md) - Single convolution operation

### Color Space

- [colormath](kernels/colormath.md) - Color space transformations
- [hsi](kernels/hsi.md) - RGB ↔ HSI color space conversions

### Demonstrations

- [demo](kernels/demo.md) - Simple array addition example
- [mandelbrot](kernels/mandelbrot.md) - Mandelbrot set computation
- [operators](kernels/operators.md) - Mathematical operators on arrays

## Interface Conventions

### Parameter Hints

All kernel interfaces use parameter hints to specify data flow and buffer management:

- **`in`**: Data copied from host to device before kernel execution
- **`out`**: Data copied from device to host after kernel execution
- **`inout`**: Data copied both directions
- **`out_like <param>`**: Allocate output buffer matching another parameter's shape/type
- **`resident`**: Buffer already on device (reduces data transfer)

### Derived Parameters

Dimensions are automatically derived from array parameters:

- **`width_of <param>`**: Extract width from 2D array
- **`height_of <param>`**: Extract height from 2D array
- **`depth_of <param>`**: Extract depth from 3D array
- **`size_of <param>`**: Total element count

### Extensions

Many kernels provide `_res` extensions for resident buffer operations:

```python
# Standard: allocates output and copies data
result = prog.median3x3(input_array)

# Resident: uses pre-allocated GPU buffers
prog.median3x3_res(input_buffer, output_buffer)
```

## Usage Example

```python
import core
import numpy as np

# Load kernel interface
prog = core.load_program("mandelbrot")

# Create input grid
grid = np.mgrid[-2:2:1024j, -2:2:1024j].astype(np.complex64)

# Execute kernel
result = prog.mandelbrot(grid, maxiter=256)
```

## Backend Support

All kernel interfaces are supported across backends:

- **OpenCL**: Full support, mature and tested
- **Metal**: Full support via template fallback
- **CUDA**: Full support (all 12+ interfaces compile and run)
- **Numba**: JIT-compiled CPU fallback

## Detailed Documentation

Click on any kernel interface above to see:

- Python signatures
- Interface specification
- Parameter explanations
- OpenCL kernel stubs
- Usage examples

## Creating Custom Interfaces

See [architecture](architecture.rst) for details on defining custom kernel interfaces using the Yapocis IDL.
