# Cython Implementation for Yapocis

## Overview

This document describes the Cython-optimized implementation of yapocis kernel wrappers, providing performance
improvements through reduced Python overhead.

## Files Created

### Core Base Classes

- `src/yapocis/core/kernel_base_cy.pyx` - Cython implementation of KernelBase
- `src/yapocis/core/kernel_base_cy.pxd` - Cython header file with type declarations

### Metal Backend

- `src/yapocis/metallic/kernel_cy.pyx` - Cython implementation of Metal kernel wrapper
- `src/yapocis/metallic/kernel_cy.pxd` - Cython header file for Metal kernel

### Build Configuration

- `setup.py` - Setuptools configuration for building Cython extensions
- `pyproject.toml` - Updated to include Cython in dev dependencies

### Testing & Benchmarks

- `claude-debug/test_cython.py` - Basic functionality test
- `claude-debug/simple_benchmark.py` - Performance comparison script

## Architecture

### Design Principles

1. **API Compatibility**: Cython versions maintain 100% API compatibility with Python versions
2. **Runtime Selection**: Environment variable `YAPOCIS_USE_CYTHON` controls which implementation is used
3. **Type Optimization**: Instance variables declared in `.pxd` files for optimal performance
4. **Method Declarations**:
    - `cpdef` methods: Accessible from both Python and Cython (slight overhead)
    - `cdef` methods: Cython-only (maximum performance, used for internal helpers)

### Optimizations

1. **Type Declarations**: C-level type declarations reduce dynamic lookups
2. **cpdef Methods**: Dual Python/Cython access with reduced call overhead
3. **Bounded Checks Off**: `boundscheck=False` eliminates array boundary checks
4. **Wraparound Off**: `wraparound=False` disables negative indexing checks
5. **C Division**: `cdivision=True` uses C-style division (faster)

## Usage

### Building Extensions

```bash
# Install Cython (already in dev dependencies)
uv sync --group dev

# Build extensions in-place
uv run python setup.py build_ext --inplace
```

### Runtime Selection

```python
import os

# Use Cython-optimized version
os.environ["YAPOCIS_USE_CYTHON"] = "1"
os.environ["YAPOCIS_BACKEND"] = "metal"

from yapocis.core import load_program

prog = load_program('mandelbrot')
# Now using Cython-optimized Metal kernel
```

```python
# Use Python version (default)
os.environ["YAPOCIS_USE_CYTHON"] = "0"  # or don't set it
os.environ["YAPOCIS_BACKEND"] = "metal"

from yapocis.core import load_program

prog = load_program('mandelbrot')
# Now using pure Python Metal kernel
```

## Performance Results

Benchmark results from `simple_benchmark.py` on Apple M1 Pro:

| Size      | Python (ms) | Cython (ms) | Speedup   |
|-----------|-------------|-------------|-----------|
| 256x256   | 0.484       | 0.527       | 0.92x     |
| 512x512   | 1.127       | 0.868       | **1.30x** |
| 1024x1024 | 2.138       | 2.197       | 0.97x     |

### Analysis

- **Best case**: 1.30x speedup at 512x512 resolution
- **Typical**: ~1.0x (comparable performance)
- **Overhead**: Small arrays (256x256) show slight overhead from Cython initialization

The modest improvements are expected for GPU-bound workloads where most time is spent in GPU execution rather than
Python overhead. The Cython optimization primarily benefits the argument preparation and marshalling phases.

## Implementation Details

### KernelBaseCy (kernel_base_cy.pyx)

Cython-optimized base class providing:

- `cpdef` methods for common operations (read, write, ensure, handle_*)
- Type-declared instance variables in `.pxd` file
- Abstract methods that subclasses must implement
- Template method pattern for `__call__()` orchestration

Key optimizations:

- `cdef int` for loop counters
- `cdef object` for Python objects that need C-level storage
- `bint` for boolean flags

### KernelCy (kernel_cy.pyx)

Metal-specific Cython kernel wrapper:

- Inherits from KernelBaseCy
- Implements Metal-specific abstract methods
- `cdef void _show_args()` helper for error reporting (C-only, maximum performance)
- `cpdef void _execute_kernel_with_args()` for kernel execution (accessible from base class)

## Future Enhancements

1. **Additional Backends**: Implement Cython versions for OpenCL and CUDA
2. **Further Optimization**: Profile hot paths and add more `cdef` methods
3. **Compile-time Specialization**: Generate specialized kernels for common array types
4. **Parallel Builds**: Use `nthreads` in cythonize for faster compilation

## Testing

```bash
# Run basic test
uv run python claude-debug/test_cython.py

# Run performance comparison
uv run python claude-debug/simple_benchmark.py

# Run standard test suite with Cython
YAPOCIS_USE_CYTHON=1 pytest tests/test_backend_smoke.py -v
```

## Notes

- Cython extensions are optional - yapocis works without them
- Build warnings about fallthrough annotations are harmless
- Extensions are platform-specific (.so on Linux/macOS, .pyd on Windows)
- Rebuilding required after modifying .pyx or .pxd files

## References

- Cython Documentation: https://cython.readthedocs.io/
- Performance Tips: https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html
- Extension Types: https://cython.readthedocs.io/en/latest/src/userguide/extension_types.html
