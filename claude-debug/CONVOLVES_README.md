# Convolves Interface: 1D Signal Processing Kernels

This document describes the implementation and usage of the `convolves` interface, which provides 58 pre-defined 1D convolution kernels for signal processing applications.

## Overview

The `convolves` interface is a template-based interface that generates multiple convolution kernels from a YAML specification. It demonstrates the power of yapocis's Mako template system for creating parameterized kernel collections.

## Files Created

### Core Implementation
- **`src/yapocis/interfaces/convolves.yml`** - YAML specification defining kernel types and sizes
- **`src/yapocis/lib/convolve_kernels.py`** - Python module for generating kernel weights
- **`tests/test_convolve_kernels.py`** - Comprehensive test suite (52 tests)
- **`claude-debug/demo_convolve_kernels.py`** - Demonstration script with visualizations

### Integration Updates
- **`src/yapocis/scripts/tools.py`** - Added convolves template rendering support
- **`src/yapocis/scripts/package.py`** - Added convolves package generation support

## Kernel Categories

### 1. Windowing Functions (18 kernels)
Used for spectral analysis, filtering, and tapering signals:
- **Hamming** (sizes: 5, 9, 13, 17, 25) - General-purpose window with low sidelobes
- **Hann** (sizes: 5, 9, 13, 17, 25) - Smooth window, good frequency resolution
- **Blackman** (sizes: 7, 11, 15, 21) - Excellent sidelobe suppression
- **Bartlett** (sizes: 5, 9, 13, 17) - Simple triangular taper

### 2. Smoothing Filters (20 kernels)
Used for noise reduction and averaging:
- **Box** (sizes: 3, 5, 7, 9, 11, 15, 21) - Uniform averaging filter
- **Triangle** (sizes: 3, 5, 7, 9, 11) - Weighted averaging with linear falloff
- **Binomial** (sizes: 3, 5, 7, 9) - Approximates Gaussian blur

### 3. Derivative Operators (8 kernels)
Used for edge detection and gradient estimation:
- **deriv1** (sizes: 3, 5, 7) - First derivative (central difference)
- **deriv2** (sizes: 3, 5, 7) - Second derivative (discrete Laplacian)
- **sobel** (size: 3) - Sobel edge detector
- **prewitt** (size: 3) - Prewitt edge detector

### 4. High-pass Filters (6 kernels)
Used for edge enhancement and sharpening:
- **sharpen** (sizes: 3, 5) - Simple sharpening kernels
- **unsharp** (sizes: 3, 5, 7) - Unsharp masking for detail enhancement

### 5. Band-pass Filters (4 kernels)
Used for feature detection:
- **dog** (sizes: 5, 7, 9, 11) - Difference of Gaussians approximation

### 6. Custom Kernels (4 kernels)
Specialized kernels for specific applications:
- **edge3** - Simple edge detection: [-1, 0, 1]
- **edge5** - Extended edge detection: [-1, -0.5, 0, 0.5, 1]
- **savgol5** - Savitzky-Golay smoothing (5-point)
- **savgol7** - Savitzky-Golay smoothing (7-point)

## Mathematical Properties

All kernels satisfy important mathematical properties verified by tests:

1. **Smoothing kernels** (box, triangle, binomial, windows): Sum to 1.0 (preserve DC/energy)
2. **Derivative kernels** (deriv1, deriv2, sobel, prewitt, dog): Sum to 0.0 (remove DC)
3. **Symmetric kernels**: Equal to their reverse (most smoothing filters)
4. **Antisymmetric kernels**: First derivatives (deriv1[i] = -deriv1[n-i])

## Usage Examples

### Basic Usage

```python
from yapocis.lib.convolve_kernels import generate_kernel
import numpy as np

# Generate a Hamming window
kernel = generate_kernel("hamming", 9)
print(f"Kernel: {kernel}")
print(f"Sum: {np.sum(kernel)}")  # Should be 1.0

# Apply to a signal
signal = np.random.randn(100)
smoothed = np.convolve(signal, kernel, mode='same')
```

### Using via Interface

```python
from yapocis.core.interface import get_interface
from yapocis.core.interface_context import load_interface_context
from yapocis.lib.convolve_kernels import generate_convs_from_spec
from mako.template import Template
from yapocis.core.interface import load_interface

# Load and render interface
interface_text = load_interface("convolves.interface")
context = load_interface_context("convolves")
convs = generate_convs_from_spec(context)
context['convs'] = convs

template = Template(interface_text)
rendered = template.render(**context)
interface = get_interface(rendered)

# List all available kernels
for kernel in interface.kernels:
    print(kernel.name)
```

### Running the Demonstration

```bash
python claude-debug/demo_convolve_kernels.py
```

This generates 5 visualizations showing:
1. **kernel_weights.png** - Visual representation of kernel coefficients
2. **smoothing_filters.png** - Comparison of different smoothing filters
3. **derivative_filters.png** - Edge detection with derivative filters
4. **frequency_response.png** - How filters affect different frequencies
5. **window_comparison.png** - Window shapes and frequency responses

## Implementation Details

### Kernel Generation

The `generate_kernel()` function in `convolve_kernels.py` provides a unified interface:

```python
kernel = generate_kernel(kernel_type, size)
```

All kernels are returned as float32 numpy arrays, properly normalized according to their type.

### Generator Functions

Each kernel type has a dedicated generator function:
- `generate_hamming(size)` - Hamming window
- `generate_box(size)` - Box/uniform filter
- `generate_deriv1(size)` - First derivative
- etc.

### Template Rendering

The interface uses Mako templates similar to `convolvesep`:

```
interface convolves {
%for name,conv in convs:
    kernel ${name}(sizeof a, in float a, outlike a);
%endfor
};
```

The `convs` list is generated at runtime from the YAML specification.

## Test Coverage

The test suite (`test_convolve_kernels.py`) includes 52 tests:

- **14 tests** for individual kernel generators
- **2 tests** for kernel registry
- **17 tests** for mathematical properties (normalization, symmetry)
- **3 tests** for YAML spec processing
- **5 tests** for signal processing applications
- **2 tests** for interface integration
- **9 tests** for kernel comparison

All tests pass:
```
tests/test_convolve_kernels.py: 52 passed
```

## Adding New Kernels

To add new kernel types:

1. **Add generator function** to `convolve_kernels.py`:
```python
def generate_my_kernel(size):
    """Generate my custom kernel."""
    weights = ...  # Your computation
    return weights.astype(np.float32)
```

2. **Register in KERNEL_GENERATORS dict**:
```python
KERNEL_GENERATORS = {
    ...
    "my_kernel": generate_my_kernel,
}
```

3. **Add to YAML spec** (`convolves.yml`):
```yaml
custom:
  - type: my_kernel
    sizes: [3, 5, 7]
```

4. **Add tests** to `test_convolve_kernels.py`

## Signal Processing Applications

### Noise Reduction
Use smoothing filters (box, hamming, binomial) to reduce random noise while preserving signal features.

### Edge Detection
Use derivative filters (deriv1, sobel) to find sharp transitions in signals.

### Feature Enhancement
Use high-pass filters (sharpen, unsharp) to enhance edges and details.

### Spectral Analysis
Use windowing functions (hamming, hann, blackman) to reduce spectral leakage in FFT operations.

### Multi-scale Processing
Use different kernel sizes to analyze features at multiple scales.

## Performance Considerations

- All kernels return float32 for GPU compatibility
- Kernels are generated on-demand (not pre-computed)
- Small kernels (size < 15) are very fast
- Large kernels may be slower but more accurate

## References

### Window Functions
- Hamming: 0.54 - 0.46*cos(2πn/(N-1))
- Hann: 0.5*(1 - cos(2πn/(N-1)))
- Blackman: a0 - a1*cos(2πn/(N-1)) + a2*cos(4πn/(N-1))

### Derivative Approximations
- 3-point: [-0.5, 0, 0.5]
- 5-point: [1, -8, 0, 8, -1]/12 (4th order accurate)

### Binomial Coefficients
- Size 3: [1, 2, 1] (from (1+1)²)
- Size 5: [1, 4, 6, 4, 1] (from (1+1)⁴)
- Size 7: [1, 6, 15, 20, 15, 6, 1] (from (1+1)⁶)

## License

Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.
Part of the yapocis project.
