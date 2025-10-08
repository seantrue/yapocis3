# Yapocis

[![PyPI version](https://badge.fury.io/py/yapocis.svg)](https://badge.fury.io/py/yapocis)
[![Python Versions](https://img.shields.io/pypi/pyversions/yapocis.svg)](https://pypi.org/project/yapocis/)
[![License](https://img.shields.io/github/license/seantrue/yapocis3.svg)](https://github.com/seantrue/yapocis3/blob/main/LICENSE)
[![CI Status](https://github.com/seantrue/yapocis3/workflows/CI/badge.svg)](https://github.com/seantrue/yapocis3/actions)

## Overview

Yapocis is a multi-backend GPU/CPU computing framework for image processing and graphics operations. It provides a
unified Python interface to OpenCL, CUDA, Metal, and Numba backends through an IDL-like interface definition language
and Mako template system.

The framework enables developers to write compute kernels once and execute them across different hardware platforms
without code modification. Backend selection occurs at runtime through environment variables or programmatic
configuration, proving that Write Once, Run Anywhere™ can actually work if you squint hard enough and control all the
variables.

For the archaeological record of how this came to be, see [HISTORY.md](HISTORY.md).

It includes easy-to-call-from-Python, reasonably performing implementations of kernels such as Mandelbrot, 3x3 median,
image gradients, HSI <-> RGB conversions, color math, 2D graphics primitives (lines, circles, rectangles, text), and
signed distance field path tracing with Jump Flood Algorithm. I use it to develop much less generally useful
implementations of bizarre numerical processing ideas.

**Version 0.3.x Development Focus:**

1. **Performance Infrastructure**: Comprehensive performance tracking system with DuckDB storage, cross-backend
   comparison, and longitudinal analysis. Because if you can't measure it, you can't optimize it. And if you can measure
   it, you'll discover it's slower than you thought.

2. **Validation Framework**: Automated regression testing with pixel-exact validation across backends.

3. **Modern Packaging**: Migration to hatchling, modern tooling (ruff, isort, mypy), and Python 3.13 support. The good
   news: Python packaging has been solved. The bad news: it was solved again. And again.

4. **Backend Maturity**: Complete feature parity across OpenCL, Metal, and Numba backends with extensive test coverage.
   Write once, debug everywhere, ship when ready.

**Future Roadmap:**

- Automated version management with setuptools_scm (eliminating the last remaining manual step)
- Pre-commit hooks for code quality enforcement (because self-discipline is overrated)
- CLI entry points for common development tools
- Backend-aware scheduling strategies for optimal hardware utilization
- Enhanced documentation with comprehensive examples (eventually)

## Key Features

### IDL-Based Interface System

Define kernel interfaces once using an intuitive interface definition language loosely inspired by Apollo NCS/OSF
DCE/Microsoft IDL, which is to say, inspired by every RPC system ever designed and none of them:

```idl
interface mandelbrot {
    kernel mandelbrot(in complex64 q, out_like short q, in int32 maxiter);
};
```

The framework automatically handles:

- Memory allocation and management (the hard parts)
- Data transfer between host and device (the slow parts)
- Type marshaling and buffer hints (the error-prone parts)
- Multi-dimensional array handling (the confusing parts)

In other words, it does everything except write the kernel for you. We're working on that. Really.

### Multi-Backend Architecture

Execute the same code across different compute platforms:

```python
import core
from core import get_manager

# Select backend programmatically
mgr = get_manager("metal")  # or "ocl", "cuda", "numba"
prog = core.load_program("mandelbrot")
result = prog.mandelbrot(input_data, maxiter=1000)

# Or use environment variable
# export YAPOCIS_BACKEND=ocl
```

**Supported Backends:**

- **OpenCL** (via PyOpenCL): Cross-platform GPU/CPU support, runs everywhere except where it doesn't
- **Metal** (via PyMetallic): Apple GPU acceleration, because some of us chose the shiny hardware
- **CUDA** (via CuPy): NVIDIA GPU support, for those who prefer green LEDs
- **Numba**: CPU fallback with JIT compilation, surprisingly fast for a "fallback"

### High-Level Graphics Libraries

Pre-built implementations for common operations:

- 2D graphics primitives (lines, circles, rectangles, text rendering in multiple fonts and rotations)
- Image processing (convolution, gradients, histogram operations)
- Color space conversions (RGB, HSI, color math)
- Anisotropic diffusion (because sometimes you need your blur to be anisotropic)
- Signed distance field path tracing
- Experimental matplotlib backend

## Installation

**Requirements:**

- Python 3.10+ (Python 3.13 recommended for development)
- NumPy 2.1+ (because NumPy 1.x is so 2020)
- Backend-specific dependencies (PyOpenCL, PyMetallic, CuPy, or Numba)
- Sufficient RAM

**Basic Installation:**

```bash
pip install yapocis
```

**Development Installation:**

```bash
# Clone repository
git clone https://github.com/seantrue/yapocis3.git
cd yapocis3

# Install with development tools (requires Python 3.13)
make install-dev

# Or manually, for those who enjoy typing
uv venv .venv --python python3.13 && source .venv/bin/activate
uv pip install -e ".[dev]"
```

**Backend-Specific Installation:**

```bash
# OpenCL
pip install yapocis[opencl]

# Metal (macOS only, sorry penguins)
pip install yapocis[metal]

# Numba (the universal adapter)
pip install yapocis[numba]

# Cuda (ubiquitous hardware)
pip install yapocis[cuda]

# All backends (very unlikely that you will get Cuda and Metal on the same machine)
pip install yapocis[all-backends]
```

- Backends: `ocl` (OpenCL via PyOpenCL), `cuda` (CUDA via PyCUDA), `metal` (PyMetallic shim), and `numba` (CPU with
  JIT).
- Select backend via env: `export YAPOCIS_BACKEND=ocl|cuda|metal|numba`.
- Select device via env: `export YAPOCIS_DEVICE=0` (or device name).
- You can also select explicitly in code:

```python
import core
from core import get_manager
import numpy as np

# Initialize backend
mgr = get_manager("ocl")  # OpenCL backend

# Load and execute kernel
prog = core.load_program("mandelbrot")
grid = np.mgrid[-2:2:1024j, -2:2:1024j].astype(np.complex64)
result = prog.mandelbrot(grid, maxiter=256)

# Admire the fractal
```

### Graphics Example

**CUDA backend notes:**

- Uses PyCUDA for CUDA kernel compilation and execution
- All 12 kernel interfaces compile and run successfully (100% compatibility)
- Includes graphics, image processing, color conversion, and SDF path tracing kernels
- Supports NVIDIA GPUs with CUDA capability
- Device selection via `YAPOCIS_DEVICE` (numeric index or device name)
- Row-major storage order matches NumPy (opposite of OpenCL/Metal)

**Numba backend notes:**

- Provides CPU fallback when GPU backends are unavailable
- Uses Numba's JIT compilation for Python/NumPy code
- Automatically selected when no GPU backend is available
- Useful for testing and development without GPU hardware

# Create graphics context

g = Graphics4()
canvas = np.zeros((512, 512, 4), dtype=np.float32)

# Draw primitives

circles = [(256, 256, 100), (128, 128, 50)]
g.circles(canvas, circles, color=[1.0, 0.0, 0.0, 1.0])

# Retrieve result

result = g.read(canvas)

# Profit (not guaranteed)

```

## Development

### Running Tests

```bash
# All tests
make test

# Run backend-specific tests
make test-ocl    # OpenCL tests
make test-cuda   # CUDA tests
make test-metal  # Metal tests
make test-numba  # Numba tests

# Code quality
make format      # Format with isort and black
make lint        # Lint with ruff
make typecheck   # Type check with mypy
make check       # Run all checks

# Building & publishing
make build           # Build distribution packages
make publish-test    # Publish to TestPyPI
make publish         # Publish to PyPI
```

### Building and Publishing

```bash
# Build distribution packages
make build

# Publish to TestPyPI (for the cautious)
make publish-test

# Publish to PyPI (for the brave)
make publish
```

## Architecture

### Core Components

**Backend Manager** (`yapocis.core.manager`)

- Device enumeration and selection
- Backend-specific kernel compilation
- Runtime backend switching (because flexibility)

**Interface Parser** (`yapocis.core.interface_parser`)

- PyParsing-based IDL parser
- Automatic parameter marshaling
- Buffer hint processing (in, out, inout, out_like, resident)

**Template System** (`yapocis.kernels/`, `yapocis.metalkernels/`, etc.)

- Mako templates for kernel code generation
- Backend-specific optimizations
- Automatic coordinate system handling (column-major vs row-major, forever debated)

**Validation Framework** (`yapocis/utils/validation.py`)

- Pixel-exact regression testing (because "close enough" isn't)
- Cross-backend output comparison
- Automated test promotion

**Performance Tracking** (`yapocis/utils/performance.py`)

- DuckDB-backed performance database
- Cross-backend comparison
- Longitudinal trend analysis
- Automated regression detection (it's not paranoia if it's real)

## Modern Infrastructure (v0.3.1+)

- **Build System**: Hatchling (PEP 517/518 compliant)
- **Development Tools**: isort, black, ruff, mypy (the modern Python stack)
- **Testing**: pytest with coverage tracking (because untested code is broken code)
- **Publishing**: Automated PyPI workflow with twine
- **Changelog**: Automated generation with git-cliff (conventional commits appreciated)
- **Python**: 3.13+ enforced in development environments (keeping up with the times)

See [RENOVATE.md](RENOVATE.md) for detailed modernization documentation.

## Documentation

- **[CLAUDE.md](CLAUDE.md)**: Development guide and architecture reference
- **[CONTRIBUTORS.md](CONTRIBUTORS.md)**: Contribution guidelines
- **[CHANGELOG.md](CHANGELOG.md)**: Release notes and version history
- **[AI.md](AI.md)**: AI-assisted development tracking (yes, really)
- **[HISTORY.md](HISTORY.md)**: Project origins and evolution (the archaeological layer)

## Contributing

Contributions are welcome. Please see [CONTRIBUTORS.md](CONTRIBUTORS.md) for guidelines.

**Development workflow:**

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality (test-driven development, not test-eventually development)
4. Follow code style guidelines (`make format`, `make lint`)
5. Use conventional commits for changelog generation
6. Submit pull request
7. Wait for CI to find the bugs you didn't

## Known Issues

There are several. Some we remember:

- Untested on Windows (pull requests welcome)
- CUDA backend less mature than others (volunteers needed)
- Documentation could always be better (but couldn't it always?)

## License

MIT License - see [LICENSE.txt](LICENSE.txt) for details.

Because sharing is caring, and restrictive licenses are so 1990s.

## Acknowledgments

- PyOpenCL community and Andreas Klöckner (for making OpenCL usable)
- Numba development team (for making Python fast)
- Apple Metal framework (for making GPUs accessible to hipsters)
- NVIDIA CUDA platform (for making GPUs ubiquitous)
- The Python packaging ecosystem (for never being boring)
