# API Reference

Auto-generated API documentation from source code docstrings.

## Package Structure

The yapocis package is organized into several sub-packages:

```{toctree}
:maxdepth: 1

core
lib
utils
assets
```

## Core Package

The `yapocis.core` module provides the main entry points:

- `load_program()` - Load a kernel program by interface name
- `get_backends()` - List available backends
- `get_manager()` - Get a backend-specific kernel manager

## Library Package

The `yapocis.lib` package contains high-level library modules that wrap kernel interfaces with convenient Python APIs.

## Backend Packages

Backend-specific implementations (not typically used directly):

- `yapocis.ocl` - OpenCL backend
- `yapocis.metallic` - Metal backend
- `yapocis.cu` - CUDA backend
- `yapocis.numba` - Numba backend
