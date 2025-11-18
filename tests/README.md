# Yapocis Test Suite

This directory contains the reorganized test suite for Yapocis, structured for easy navigation and efficient test execution.

## Organization Principles

- **Primary organization**: By yapocis module/component (lib.graphics, lib.fft, etc.)
- **Secondary organization**: Separate correctness, regression, performance, and backend-specific tests
- **Maximum nesting**: 3 levels deep (`tests/category/subcategory/test_*.py`)
- **Smoke tests**: Marker-based (`@pytest.mark.smoke`) - at least one per library module

## Directory Structure

```
tests/
├── README.md                    # This file
├── conftest.py                  # Fixtures, markers, auto-use fixtures
├── __init__.py                  # Shared test utilities
├── random_data.py               # Test data generators
│
├── backend/                     # Backend infrastructure tests
│   ├── test_smoke.py           # Compile all interfaces (@smoke)
│   └── test_runtime.py         # Basic execution per backend (@smoke)
│
├── lib/                         # Library component tests
│   ├── graphics/               # Graphics primitives
│   │   ├── test_graphics.py            # Graphics1 correctness (@smoke)
│   │   ├── test_graphics4.py           # Graphics4 correctness (@smoke)
│   │   ├── test_antialiasing.py        # AA functionality
│   │   ├── test_kerneldraw.py          # KernelDraw1
│   │   ├── test_kerneldraw3.py         # KernelDraw3
│   │   └── test_kerneldraw4.py         # KernelDraw4
│   │
│   ├── mplbackend/             # Matplotlib backend tests
│   │   ├── test_backend.py             # Core mplbackend functionality
│   │   ├── test_pyplot.py              # Pyplot adaptation tests
│   │   ├── test_images.py              # Image integration tests
│   │   ├── test_advanced.py            # Advanced rendering scenarios
│   │   ├── test_rendering.py           # Rendering tests
│   │   ├── test_comparison.py          # Backend comparison
│   │   └── test_math_comparison.py     # Math rendering comparison
│   │
│   ├── processing/             # Image processing
│   │   ├── test_convolve.py            # Convolution ops (@smoke)
│   │   ├── test_convolve_kernels.py    # Kernel-level convolve
│   │   ├── test_anisodiff.py           # Anisotropic diffusion (@smoke)
│   │   ├── test_colormath.py           # Color space conversions (@smoke)
│   │   ├── test_imageops.py            # General image operations (@smoke)
│   │   └── test_median.py              # Median filtering (@smoke)
│   │
│   ├── compute/                # Computational kernels
│   │   ├── test_mandelbrot.py          # Mandelbrot fractals (@smoke)
│   │   ├── test_operators.py           # Basic operators (@smoke)
│   │   ├── test_gaussians.py           # Gaussian operations (@smoke)
│   │   ├── test_truetrace.py           # SDF path tracing (@smoke)
│   │   ├── test_gradient.py            # Gradient computation (@smoke)
│   │   ├── test_zc.py                  # Zero crossing (@smoke)
│   │   └── test_hires.py               # High-res operations
│   │
│   ├── fft/                    # FFT operations
│   │   ├── test_fft.py                 # FFT kernels (@smoke)
│   │   └── test_fft_library.py         # FFT library interface
│   │
│   └── neural/                 # Neural networks
│       ├── test_cnn.py                 # CNN models (@smoke)
│       ├── test_cnn_kernels.py         # CNN kernel primitives
│       └── test_modern_cnn.py          # Modern architectures (VGG)
│
├── kernelpack/                  # Compiled kernel packages
│   ├── test_comprehensive.py   # General kernelpack tests (@smoke)
│   └── test_convolvesep.py     # Specific convolvesep package
│
├── async/                       # Async execution
│   ├── test_execution.py       # Async patterns (@smoke)
│   ├── test_modes.py           # Async mode variations
│   └── test_performance.py     # Async performance (@performance)
│
├── performance/                 # Performance benchmarks
│   ├── test_kernel_perf.py     # Raw kernel performance (@performance)
│   ├── test_library_perf.py    # Library-level performance (@performance)
│   └── README.md               # Performance testing guide
│
├── regression/                  # Regression/validation
│   ├── test_validation.py      # Validation framework examples (@validation)
│   └── test_pixel_exact.py     # Pixel-perfect graphics validation (@validation, @smoke)
│
└── integration/                 # System integration
    ├── test_config.py          # Configuration management (@smoke)
    ├── test_tools_cli.py       # CLI tools (@smoke)
    ├── test_registries.py      # Module discovery
    ├── test_examples.py        # Example scripts
    ├── test_utils.py           # Utility functions
    └── test_rpc.py             # RPC communication
```

## Test Markers

### Primary Markers

- **`@pytest.mark.smoke`** - Quick validation tests (~20 tests, runs in 1-2 minutes)
  - At least one test per library module
  - Tests basic functionality with small data
  - Runs on all backends (unless backend-specific)
  - Ideal for CI pre-commit checks

- **`@pytest.mark.validation`** - Regression validation tests
  - Compares outputs against stored reference data
  - Detects unintended behavior changes
  - Run with `--validate` flag

- **`@pytest.mark.performance`** - Performance benchmark tests
  - Tracks execution time and memory usage
  - Not run by default (opt-in with `-m performance`)
  - Results stored in `performance/` directory

- **`@pytest.mark.track_memory`** - Memory usage tracking
  - Monitors peak memory consumption
  - Useful for detecting memory leaks

### Backend Markers

- **`@pytest.mark.ocl`** - OpenCL-specific tests
- **`@pytest.mark.metal`** - Metal-specific tests
- **`@pytest.mark.cuda`** - CUDA-specific tests
- **`@pytest.mark.numba`** - Numba-specific tests

Tests are automatically skipped if the required backend is not available.

## Running Tests

### Quick Smoke Test (Recommended for Development)

```bash
# Run all smoke tests (~20 tests, 1-2 minutes)
pytest -m smoke tests/

# Smoke tests for specific backend
pytest -m "smoke and metal" tests/
pytest -m "smoke and ocl" tests/
```

### By Category

```bash
# All backend infrastructure tests
pytest tests/backend/

# All graphics library tests
pytest tests/lib/graphics/

# All processing library tests
pytest tests/lib/processing/

# All integration tests
pytest tests/integration/
```

### By Test Type

```bash
# All tests (except performance)
pytest tests/

# Performance benchmarks only
pytest -m performance tests/

# Validation/regression tests only
pytest -m validation tests/

# All tests except performance
pytest -m "not performance" tests/
```

### Specific Tests

```bash
# Single test file
pytest tests/lib/graphics/test_graphics.py

# Single test function
pytest tests/lib/graphics/test_graphics.py::test_smoke_line

# All tests matching pattern
pytest -k "convolve" tests/
```

### With Validation

```bash
# Run with validation checks
pytest --validate tests/

# Update validation references
pytest --update-validation tests/

# Set validation tolerance
pytest --validate --validation-tolerance=1e-5 tests/
```

### Backend-Specific

```bash
# Only OpenCL tests
pytest -m ocl tests/

# Metal tests with smoke suite
pytest -m "smoke and metal" tests/

# All backends for specific component
YAPOCIS_BACKEND=ocl pytest tests/lib/fft/
YAPOCIS_BACKEND=metal pytest tests/lib/fft/
```

## Smoke Test Coverage

The smoke test suite provides quick validation of all major yapocis components:

### Backend Infrastructure (2 tests)
- Interface compilation check
- Basic kernel execution

### Library Modules (15 tests - one per module)
1. **graphics** - Basic line drawing
2. **graphics4** - Lines4 operation
3. **convolve** - 1D convolution
4. **anisodiff** - Basic diffusion
5. **colormath** - RGB to HSI conversion
6. **imageops** - Basic image operation
7. **median** - Median3x3 filter
8. **mandelbrot** - Mandelbrot set computation
9. **operators** - Add/multiply operations
10. **gaussian** - Gaussian blur
11. **truetrace** - SDF tracing
12. **gradient** - Gradient computation
13. **zc** - Zero crossing detection
14. **fft** - 1D FFT forward
15. **cnn** - Basic conv2d layer

### Infrastructure (5 tests)
16. **kernelpack** - Load compiled package
17. **async** - Basic async execution
18. **validation** - Pixel-exact validation
19. **config** - Device selection
20. **cli_tools** - CLI functionality

**Total: ~22 smoke tests covering all library modules + infrastructure**

## Smoke Test Design Guidelines

Each smoke test should:

1. **Execute quickly** - Target <5 seconds per test
2. **Use small data** - 256x256 images, not 4096x4096
3. **Test basic functionality** - Common use case, not edge cases
4. **Run on all backends** - Unless backend-specific
5. **Fail fast** - Catch compilation or execution errors immediately
6. **Be deterministic** - Same inputs should produce same outputs

Example smoke test pattern:

```python
import pytest
import numpy as np
from yapocis.core import get_backends
from yapocis.lib import some_module

@pytest.mark.smoke
@pytest.mark.parametrize("backend", get_backends())
def test_smoke_operation(backend):
    """Quick smoke test for operation - ensures basic functionality works."""
    os.environ["YAPOCIS_BACKEND"] = backend

    # Small test data
    data = np.random.rand(256, 256).astype(np.float32)

    # Basic operation
    result = some_module.operation(data)

    # Simple validation
    assert result.shape == data.shape
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))
```

## CI/CD Integration

Recommended CI workflow:

```yaml
# On every commit/PR
- name: Run smoke tests
  run: pytest -m smoke tests/

# On merge to main
- name: Run full test suite
  run: pytest -m "not performance" tests/

# Nightly or weekly
- name: Run performance benchmarks
  run: pytest -m performance tests/
```

## Adding New Tests

### For a New Library Module

1. Create test file in appropriate `lib/` subdirectory
2. Add at least one `@pytest.mark.smoke` test
3. Test across all backends with `@pytest.mark.parametrize("backend", get_backends())`
4. Add validation tests if applicable

### For a New Feature

1. Add tests to existing component file or create new file
2. Follow naming convention: `test_<feature>.py`
3. Add smoke test if it's a core feature
4. Consider backend-specific behavior

### For Performance Tests

1. Add to `performance/` directory
2. Use `@pytest.mark.performance` marker
3. Use `perf_tracker` fixture for timing
4. Document expected performance characteristics

## Performance Testing

Performance tests are opt-in and not run by default:

```bash
# Run performance benchmarks
pytest -m performance tests/

# With backend comparison
pytest -m performance tests/ --backend-comparison

# Single performance test
pytest tests/performance/test_kernel_perf.py
```

See `performance/README.md` for detailed performance testing documentation.

## Validation Framework

The validation system enables regression testing against stored reference data:

```bash
# List validation references
yapocis-validate list

# Promote test output to reference
yapocis-validate promote "test_name"

# Run with validation
pytest --validate tests/

# Update all validation references
pytest --update-validation tests/
```

## Troubleshooting

### Tests are being skipped

- Check that required backend dependencies are installed
- Use `pytest -v` to see skip reasons
- Install backend: `uv sync --extra <backend> --group dev`

### Smoke tests are slow

- Check data sizes (should be small, e.g., 256x256)
- Ensure not using high-resolution images
- Check for unnecessary file I/O

### Performance tests not running

- Performance tests require explicit opt-in: `pytest -m performance`
- This is by design to keep regular test runs fast

### Backend-specific failures

- Set `YAPOCIS_BACKEND` to isolate: `YAPOCIS_BACKEND=ocl pytest tests/`
- Check backend-specific markers and skip conditions
- Review backend-specific template differences

## Migration from Old tests/ Directory

The old `tests/` directory has been reorganized into this structure. Key changes:

- **Flat structure → organized by module** - Tests grouped by yapocis component
- **No smoke tests → comprehensive smoke suite** - Fast validation of all modules
- **Mixed concerns → clear separation** - Performance, correctness, backend tests separated
- **Hard to navigate → easy to find** - Logical directory structure

Old test files have been migrated to their corresponding new locations. Import paths and test names remain largely unchanged.
