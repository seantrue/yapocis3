# Compiled Kernel Packages

## Overview

The kernel package system eliminates runtime interpretation overhead by pre-compiling Python classes from interface specifications. This provides ~20-25% performance improvement for kernel calls.

## Usage

### Enable via Environment Variable (Recommended)

```bash
export YAPOCIS_KERNELPACK=1  # or 'true' or 'yes'
```

This automatically enables kernel packages for all compatible operations with automatic fallback if packages are unavailable.

### Enable via Code

```python
from yapocis.lib.gaussian import Gaussian

# Explicit opt-in (overrides environment variable)
g = Gaussian(use_package=True)

# Explicit opt-out (overrides environment variable)
g = Gaussian(use_package=False)

# Automatic (respects YAPOCIS_KERNELPACK env var)
g = Gaussian()
```

## How It Works

### Traditional Path (Runtime Interpretation)

Every kernel call interprets parameter hints at runtime:
- Parse `in`, `out`, `out_like`, `resident` qualifiers
- Determine buffer allocations
- Convert types
- Derive dimensions from `width_of`, `height_of`

### Compiled Path (Pre-Compiled)

Interface specs are compiled into Python classes with explicit operations:
- All buffer management pre-compiled
- Direct kernel calls with no interpretation
- Zero overhead from parameter analysis

Example: `yapocis/kernelpack/ocl/convolvesep.py`

### Automatic Fallback

If `YAPOCIS_KERNELPACK=1` but packages unavailable:
1. Attempts to import `yapocis.kernelpack`
2. If ImportError, silently falls back to traditional path
3. Application continues working normally

## Performance

Benchmark: Gaussian blur on 512x512 image (100 iterations)

| Method | Time per call | Speedup |
|--------|---------------|---------|
| Traditional | 2.504 ms | 1.00x |
| Compiled Package | 1.991 ms | **1.26x** |

**Time saved**: 0.513 ms per kernel call (25.8% improvement)

## Currently Implemented

- **convolvesep**: Separable convolution operations
  - Used by `Gaussian` class for blur operations
  - Supports all variants: `gauss()`, `gauss_res()`, `gauss_h()`, `gauss_v()`

## Directory Structure

```
yapocis/kernelpack/
├── __init__.py                    # Package loader
├── ocl/
│   ├── __init__.py
│   └── convolvesep.py            # OpenCL implementation
├── metal/
│   ├── __init__.py
│   └── convolvesep.py            # Metal implementation (TODO)
├── cuda/
│   ├── __init__.py
│   └── convolvesep.py            # CUDA implementation (TODO)
└── numba/
    ├── __init__.py
    └── convolvesep.py            # Numba implementation (TODO)
```

## Testing

```bash
# Test environment variable control
python claude-debug/test_kernelpack_env.py

# Test automatic fallback
python claude-debug/test_kernelpack_fallback.py

# Test integration with Gaussian
python claude-debug/test_gaussian_with_package.py

# Benchmark performance
YAPOCIS_BACKEND=ocl YAPOCIS_KERNELPACK=1 python claude-debug/benchmark_gaussian_package.py
```

## Backward Compatibility

- **Zero breaking changes**: All existing code works unchanged
- **Opt-in**: Disabled by default, must explicitly enable
- **Graceful degradation**: Falls back if packages unavailable
- **API compatible**: Wrapper classes mimic original `Kernel` interface

## Future Work

- Generate packages automatically from interface specs
- Implement for other backends (Metal, CUDA, Numba)
- Extend to other interfaces (graphics4, convolve, etc.)
- Add code generator tool: `yapocis-package <interface> --backend <backend>`
