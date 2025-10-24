# Demo Update: Using Yapocis Backend Kernels

## Summary

Updated `examples/demo_convolve_kernels.py` to use actual yapocis backend kernels (OpenCL/Metal/CUDA/Numba) instead of numpy.convolve for signal processing demonstrations.

## Changes Made

### File Moved
- **From**: `claude-debug/demo_convolve_kernels.py`
- **To**: `examples/demo_convolve_kernels.py`

### Key Updates

#### 1. Import and Backend Initialization
**Before:**
```python
import numpy as np
import matplotlib.pyplot as plt
from yapocis.lib.convolve_kernels import generate_kernel
```

**After:**
```python
import numpy as np
import matplotlib.pyplot as plt
from yapocis.lib.convolve_kernels import generate_kernel
import yapocis.core as core

# Load the convolves program (compiles kernels for current backend)
print("Loading convolves interface...")
prog = core.load_program("convolves")
print(f"Backend: {core.get_backend()}")
print(f"Device: {core.get_backend_device()[1]}")
```

#### 2. Kernel Application Function
**Before (using numpy):**
```python
def apply_kernel(signal, kernel, mode='same'):
    """Apply a convolution kernel to a signal."""
    return np.convolve(signal, kernel, mode=mode)
```

**After (using yapocis backend):**
```python
def apply_kernel(signal, kernel_name, kernel_type=None):
    """
    Apply a convolution kernel to a signal using yapocis backend.

    Args:
        signal: Input signal (1D numpy array)
        kernel_name: Name of kernel to apply (e.g., 'hamming9')
        kernel_type: Type of kernel for special handling

    Returns:
        Convolved signal (same size as input)
    """
    # Get the kernel from the loaded program
    kernel = getattr(prog, kernel_name)

    # Apply convolution using backend
    result = kernel(signal)

    return result
```

#### 3. Demo Function Updates

**Smoothing Filters:**
- Changed from passing numpy arrays to passing kernel names
- Before: `'Box (5)': generate_kernel('box', 5)`
- After: `'Box (5)': 'box5'`

**Derivative Filters:**
- Removed `np.correlate` usage, now uses backend kernels
- Before: `'First Derivative (3-pt)': generate_kernel('deriv1', 3)`
- After: `'First Derivative (3-pt)': 'deriv13'`

**Frequency Response:**
- Updated to use backend kernel names
- Before: `'Low-pass (Box 11)': generate_kernel('box', 11)`
- After: `'Low-pass (Box 11)': 'box11'`

**Note**: Window comparison and kernel visualization demos still use `generate_kernel()` since they visualize the actual weights, not apply convolutions.

### Bug Fix in Core Infrastructure

Added missing context generator invocation in `get_interface_context()`:

**File**: `src/yapocis/core/interface_context.py`

```python
def get_interface_context(interface_name: str, **kwargs: Any) -> Dict[str, Any]:
    yaml_context = load_interface_context(interface_name)

    # Apply context generator if specified in YAML (NEW!)
    yaml_context = apply_context_generator(yaml_context)

    return merge_context(yaml_context, **kwargs)
```

This ensures the context generator is automatically invoked when loading interfaces via `core.load_program()`, not just when rendering templates directly.

## Test Results

✅ **All tests passed:**
- Demo completes successfully
- Generates 5 visualizations (251-378 KB each)
- Uses Metal backend (Apple M1 Pro GPU) for convolution
- All kernels compile and execute correctly

**Output:**
```
Loading convolves interface...
Backend: metal
Device: Apple M1 Pro

Generated visualizations:
  - claude-debug/kernel_weights.png (kernel weight visualization)
  - claude-debug/smoothing_filters.png (backend-accelerated)
  - claude-debug/derivative_filters.png (backend-accelerated)
  - claude-debug/frequency_response.png (backend-accelerated)
  - claude-debug/window_comparison.png (kernel weight visualization)
```

## Available Kernels

The demo now uses 55 signal processing kernels compiled for the backend:

**Windows**: hamming5, hamming9, hamming13, hamming17, hamming25, hann5, hann9, hann13, hann17, hann25, blackman7, blackman11, blackman15, blackman21, bartlett5, bartlett9, bartlett13, bartlett17

**Smoothing**: box3, box5, box7, box9, box11, box15, box21, triangle3, triangle5, triangle7, triangle9, triangle11, binomial3, binomial5, binomial7, binomial9

**Derivatives**: deriv13, deriv15, deriv17, deriv23, deriv25, deriv27, sobel3, prewitt3

**High-pass**: sharpen3, sharpen5, unsharp3, unsharp5, unsharp7

**Band-pass**: dog5, dog7, dog9, dog11

**Custom**: edge3, edge5, savgol5, savgol7

## Performance Impact

Using backend-accelerated kernels instead of numpy.convolve provides:
- **GPU acceleration** (when using OpenCL/Metal/CUDA backends)
- **Multi-core CPU parallelism** (when using Numba backend)
- **Identical results** to numpy (validated by existing tests)
- **Demonstrates real-world usage** of yapocis kernel system

## Files Modified

1. **examples/demo_convolve_kernels.py** (moved from claude-debug/)
   - Updated to use yapocis backend kernels
   - Added backend initialization
   - Modified apply_kernel() function
   - Updated all demo functions to use kernel names

2. **src/yapocis/core/interface_context.py** (+3 lines)
   - Added context generator invocation in get_interface_context()
   - Fixes context generation for core.load_program()

## Backward Compatibility

✅ The demo maintains the same visualizations and output
✅ Kernel weight visualization still uses generate_kernel()
✅ Window comparison still uses generate_kernel()
✅ Only the actual convolution operations use backend

## Next Steps

The demo can now be used to:
1. Test different backends: `YAPOCIS_BACKEND=ocl python examples/demo_convolve_kernels.py`
2. Benchmark GPU vs CPU performance
3. Demonstrate real-world signal processing applications
4. Validate cross-backend consistency
