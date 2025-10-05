# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Yapocis is a multi-backend GPU/CPU computing framework for image processing and graphics operations. It provides a unified Python interface to OpenCL, CUDA, Metal, and Numba backends through an IDL-like interface definition language and Mako template system.

## Key Commands

### Environment Setup
```bash
# Using make (recommended)
make install-dev

# Or manually with uv
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Install with all backends
make install-backends
# Or: uv pip install -e ".[all-backends,dev]"
```

### Code Quality & Formatting
```bash
# Format code with isort and black
make format

# Check formatting
make format-check

# Lint with ruff
make lint
make lint-fix  # Auto-fix issues

# Type check with mypy
make typecheck

# Run all checks
make check

# Auto-fix all issues
make fix

# Check license compliance
make license-check
```

### Testing
```bash
# Run all tests across available backends
make test

# Run backend-specific tests
make test-ocl      # OpenCL tests
make test-metal    # Metal tests  
make test-numba    # Numba tests

# Run specific test patterns
pytest -q -k "test_graphics"
pytest -q tests/test_backend_smoke.py  # Compile all interfaces
pytest -q tests/test_backend_runtime.py  # Basic runtime sanity

# Backend markers for selective testing
pytest -q -m ocl    # Only OpenCL tests
pytest -q -m metal  # Only Metal tests
pytest -q -m numba  # Only Numba tests

# Run single test
pytest tests/test_graphics_pixel_exact.py::test_graphics_pixel_exact_backend -v
```

### Benchmarking
```bash
# Separable Gaussian convolution benchmark
python scripts/bench_convolvesep_numba.py --sizes 512,1024 --repeats 3 --backends auto

# Median 3x3 benchmark
python scripts/bench_median3x3.py --sizes 512,1024 --backends auto
```

### Changelog Management
```bash
# Generate CHANGELOG.md from git history
make changelog

# Show unreleased changes
make changelog-unreleased
```

### Building & Distribution
```bash
# Build distribution packages
make build

# Build wheel only
make build-wheel

# Publish to TestPyPI
make publish-test

# Publish to PyPI
make publish
```

### Asset Configuration
```bash
# Enable external assets (icons, images, interfaces)
export YAPOCIS_PROJECT=external
```

## Architecture Overview

### Core Components

**Backend Management (`yapocis.core.manager`)**
- `get_backends()`: Detect available backends (ocl, metal, cuda, numba)
- `get_manager(backend)`: Get backend-specific kernel manager
- Backends controlled by `YAPOCIS_BACKEND` environment variable

**Interface Definition Language**
- IDL-like syntax in `yapocis/interfaces/*.interface` files
- Parsed by `yapocis.core.interface_parser` (PyParsing-based)
- Key keywords: `in`, `out`, `inout`, `out_like`, `resident`, `width_of`, `height_of`
- Extensions allow alternate signatures (e.g., `res` for resident buffers)

**Template System**
- Mako templates per backend: `yapocis/kernels/*.mako` (OpenCL), `yapocis/metalkernels/*.mako` (Metal), `yapocis/cukernels/*.mako` (CUDA)
- Backend-specific KernelManagers resolve and compile templates
- Template resolution via `yapocis.core.mako_utils`

**High-Level Libraries (`yapocis.lib.*`)**
- `graphics.py`, `graphics4.py`: 2D graphics primitives (lines, circles, text)
- `convolve.py`: Image convolution operations
- `colormath.py`, `hsi.py`: Color space conversions
- `anisodiff.py`: Anisotropic diffusion
- `truetrace.py`: Signed distance field path tracing

### Backend Architecture

Each backend implements:
- **Device Management**: Enumerate and select compute devices
- **Buffer Management**: Allocate/manage GPU/CPU memory 
- **Kernel Management**: Compile and execute kernels with unified `load_program`/`import_program` API

**OpenCL Backend** (`yapocis.ocl.*`)
- Via PyOpenCL, supports CPU/GPU devices
- 2D tiled dispatch for large arrays

**Metal Backend** (`yapocis.metallic.*`)
- Via pymetallic shim, Apple GPU only
- Falls back to OpenCL templates when Metal-specific unavailable

**CUDA Backend** (`yapocis.cu.*`)
- Via PyCUDA, compiles through NVCC
- Template system adapted from OpenCL

**Numba Backend** (`yapocis.numba.*`)
- Pure Python/NumPy with JIT compilation
- Used for CPU fallback and testing

## Environment Variables

```bash
YAPOCIS_BACKEND=ocl|metal|cuda|numba    # Backend selection
YAPOCIS_ENGINE=<device_name_or_index>   # Device selection
YAPOCIS_CACHE=<directory>               # Kernel cache directory
YAPOCIS_TUNECACHE=<json_file>           # Autotune cache (default: ~/.cache/yapocis/tuning.json)
YAPOCIS_PROJECT=<directory>             # Additional asset root (e.g., "external")
YAPOCIS_VIEWER=<command>                # Image viewer for debugging
```

## Testing Architecture

**Backend Coverage**
- Tests automatically skip when backend dependencies unavailable
- Pytest markers: `@pytest.mark.ocl`, `@pytest.mark.metal`, etc.
- Cross-backend validation via `test_backend_crosscheck.py`

**Key Test Files**
- `test_backend_smoke.py`: Compile all interfaces across backends
- `test_backend_runtime.py`: Basic execution sanity checks
- `test_graphics_pixel_exact.py`: Pixel-perfect validation (most important for graphics)
- `test_backend_crosscheck.py`: Cross-backend output comparison

**Asset Testing**
- Set `YAPOCIS_PROJECT=external` to access full test asset library
- Icons, images, and additional interfaces in `external/` directory

## Interface Definition Language

```idl
interface mandelbrot {
    kernel mandelbrot(in complex64 q, out_like short q, int maxiter);
};

interface tiled graphics4 {
    kernel lines(width_of a, height_of a, inout float a, width_of lines, in float lines, in float distance, in float rcolor);
    extension res as lines(width_of a, height_of a, resident float a, width_of lines, in float lines, in float distance, in float rcolor);
};
```

**Parameter Hints**
- `in`, `out`, `inout`: Data flow direction
- `out_like <param>`: Allocate output with same shape as parameter
- `resident`: Use buffer in-place (reduce copies)
- `width_of`, `height_of`: Auto-derive dimensions
- `sizeof`: Auto-derive buffer sizes

## Common Development Patterns

**Loading and Using Kernels**
```python
from yapocis.core.manager import get_manager
mgr = get_manager("ocl")  # or honor YAPOCIS_BACKEND
prog = mgr.load_program("mandelbrot")
result = prog.mandelbrot(input_array, maxiter=1000)
```

**High-Level Graphics**
```python
from yapocis.lib.graphics4 import Graphics4
g = Graphics4()
canvas = np.zeros((height, width, 4), dtype=np.float32)
g.circles(canvas, [(x, y, radius)], color=[r, g, b, a])
result = g.read(canvas)
```

**Backend Testing**
```python
@pytest.mark.parametrize("backend", get_backends())
def test_operation(backend):
    os.environ["YAPOCIS_BACKEND"] = backend
    # Test implementation
```

## Asset Management

Asset lookup searches: `[yapocis_package, cwd, YAPOCIS_PROJECT]`
- Icons: `yapocis.icons.icon(name, size=64)`
- Images: `yapocis.images.image(name)`  
- Interfaces: `yapocis.interfaces.find_interface(name)`
- Kernels: Resolved per backend via AssetManager

## Validation Framework

The validation system uses stored test results for regression testing:

### Test Output Storage
- All `show_array()` calls automatically save arrays to `queue/raw/*.npz`
- Both raw arrays (.npz) and rendered images (.png) are saved
- Files named with format: `"test_name [backend].npz"`

### Validation Commands
```bash
# Manage validation references
python scripts/manage_validation.py info              # System status
python scripts/manage_validation.py list             # List references  
python scripts/manage_validation.py promote "test"   # Promote test result
python scripts/manage_validation.py promote all      # Bulk promote

# Run tests with validation
pytest --validate                    # Compare against references
pytest --update-validation          # Update references with current results
pytest --validation-tolerance=1e-5  # Set comparison tolerance
pytest -m validation                # Only validation-marked tests
```

### Test Integration
```python
@pytest.mark.validation
@pytest.mark.parametrize("backend", get_backends())
def test_operation(backend, validate_result):
    os.environ["YAPOCIS_BACKEND"] = backend
    result = perform_operation()
    validate_result("operation_name", result, backend)
```

### Validation Workflow
1. **Run tests** to generate `queue/raw/*.npz` files
2. **Review outputs** and promote good results to validation references
3. **Daily development** with `pytest --validate` to catch regressions
4. **Update references** with `--update-validation` for intentional changes