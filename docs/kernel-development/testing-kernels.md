# Testing Kernels

Comprehensive testing ensures kernels work correctly across all backends.

## Test Structure

```
tests/
├── test_backend_smoke.py       # Compile all interfaces
├── test_backend_runtime.py     # Basic execution tests
├── test_backend_crosscheck.py  # Cross-backend comparison
└── test_<kernel>_*.py          # Kernel-specific tests
```

## Basic Kernel Test

```python
import numpy as np
import pytest
from yapocis.core.manager import get_backends
from yapocis.core import manager

@pytest.mark.parametrize("backend", get_backends())
def test_gradient(backend):
    # Create test data
    img = np.random.rand(256, 256).astype(np.float32)

    # Load and run kernel
    prog = manager.load_program("gradient", backend=backend)
    grad, theta = prog.gradient(img, reach=1)

    # Verify output shape and type
    assert grad.shape == img.shape
    assert grad.dtype == np.float32
    assert not np.any(np.isnan(grad))
```

## Cross-Backend Comparison

Ensure all backends produce consistent results:

```python
@pytest.mark.parametrize("backend", get_backends())
def test_gradient_cross_backend(backend):
    img = np.random.rand(256, 256).astype(np.float32)

    prog = manager.load_program("gradient", backend=backend)
    grad, theta = prog.gradient(img, reach=1)

    # Compare against NumPy reference implementation
    expected_dx = np.gradient(img, axis=1)
    expected_dy = np.gradient(img, axis=0)
    expected = np.sqrt(expected_dx**2 + expected_dy**2)

    np.testing.assert_allclose(grad, expected, rtol=1e-4)
```

## Backend Parity Test

Compare all backends against each other:

```python
def test_backend_parity():
    img = np.random.rand(256, 256).astype(np.float32)
    results = {}

    for backend in get_backends():
        prog = manager.load_program("gradient", backend=backend)
        grad, _ = prog.gradient(img, reach=1)
        results[backend] = grad

    # All backends should match
    reference = results[list(results.keys())[0]]
    for backend, result in results.items():
        np.testing.assert_allclose(
            result, reference, rtol=1e-4,
            err_msg=f"{backend} differs from reference"
        )
```

## Smoke Tests

Verify all interfaces compile without running:

```python {.no-exec}
@pytest.mark.parametrize("backend", get_backends())
@pytest.mark.parametrize("interface", get_all_interfaces())
def test_compile(backend, interface):
    """All interfaces should compile without error."""
    prog = manager.load_program(interface, backend=backend)
    assert prog is not None
```

## Edge Cases

```python
def test_empty_input():
    prog = manager.load_program("gradient")
    img = np.array([], dtype=np.float32).reshape(0, 0)
    # Should handle gracefully
    grad, theta = prog.gradient(img, reach=1)
    assert grad.shape == (0, 0)

def test_single_pixel():
    prog = manager.load_program("gradient")
    img = np.array([[0.5]], dtype=np.float32)
    grad, theta = prog.gradient(img, reach=1)
    assert grad[0, 0] == 0.0  # No gradient for single pixel

def test_large_array():
    prog = manager.load_program("gradient")
    img = np.random.rand(4096, 4096).astype(np.float32)
    grad, theta = prog.gradient(img, reach=1)
    assert grad.shape == img.shape
```

## Image-Based Tests

```python
from yapocis.utils.basic import get_test_image

def test_with_real_image():
    img = get_test_image()

    prog = manager.load_program("gradient")
    grad, theta = prog.gradient(img, reach=1)

    # Check output properties
    assert grad.shape == img.shape[:2]
    assert 0 <= grad.min()
    assert grad.max() <= 2.0  # Reasonable gradient range
```

## Validation Framework

Yapocis includes pixel-exact validation:

```python
@pytest.mark.validation
def test_graphics_exact(validate_result):
    from yapocis.lib.graphics4 import Graphics4

    canvas = np.zeros((512, 512, 4), dtype=np.float32)
    g = Graphics4()
    g.circles(canvas, [(256, 256, 100)], color=[1, 0, 0, 1])

    # Compare against stored reference
    validate_result("graphics_circles", g.read(canvas))
```

### Managing Validation References

```bash
# View validation status
yapocis-validate info

# List pending validations
yapocis-validate list

# Promote test result to reference
yapocis-validate promote "test_name"

# Promote all pending
yapocis-validate promote all
```

## Backend Markers

Use pytest markers for selective testing:

```python
import pytest

@pytest.mark.ocl
def test_opencl_specific():
    """Only runs with OpenCL backend."""
    prog = manager.load_program("gradient", backend="ocl")
    # ...

@pytest.mark.metal
def test_metal_specific():
    """Only runs with Metal backend."""
    prog = manager.load_program("gradient", backend="metal")
    # ...

@pytest.mark.numba
def test_numba_specific():
    """Only runs with Numba backend."""
    prog = manager.load_program("gradient", backend="numba")
    # ...
```

Run specific backends:

```bash
pytest -m ocl      # OpenCL only
pytest -m metal    # Metal only
pytest -m numba    # Numba only
pytest -m cuda     # CUDA only
```

## Debugging Test Failures

### Inspect Intermediate Values

```python
from yapocis.utils.basic import show_array

def test_debug():
    img = get_test_image()
    prog = manager.load_program("gradient")
    grad, theta = prog.gradient(img, reach=1)

    # Save for visual inspection
    show_array(grad, "debug_gradient")
    show_array(theta, "debug_theta")
```

Files are saved to `queue/raw/`.

### Check Data Quality

```bash
# Analyze test output files
yapocis-queue queue/raw

# Show only files with NaN values
yapocis-queue --nan-only

# Filter by pattern
yapocis-queue --filter "gradient"
```

### Data Integrity Test

```python
def test_data_transfer():
    """Verify data round-trip integrity."""
    input_data = np.arange(100).astype(np.float32)

    # Use identity kernel
    prog = manager.load_program("demo")
    result = prog.sum(input_data, np.zeros_like(input_data))

    np.testing.assert_array_equal(result, input_data)
```

## Performance Testing

```python
import time

def test_performance():
    img = np.random.rand(1024, 1024).astype(np.float32)
    prog = manager.load_program("gradient")

    # Warmup
    prog.gradient(img, reach=1)

    # Timed runs
    times = []
    for _ in range(10):
        start = time.perf_counter()
        prog.gradient(img, reach=1)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times)
    assert avg_time < 0.1  # Should complete in 100ms
```

## Test Utilities

### Comparing Arrays

```python {.no-exec}
# Exact match
np.testing.assert_array_equal(result, expected)

# Approximate match (floating point)
np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)

# Using pytest.approx
assert result[0, 0] == pytest.approx(expected_value, rel=1e-5)
```

### Generating Test Data

```python
# Uniform random
img = np.random.rand(256, 256).astype(np.float32)

# Known pattern
img = np.zeros((256, 256), dtype=np.float32)
img[100:150, 100:150] = 1.0  # White square

# Gradient pattern
x = np.linspace(0, 1, 256)
y = np.linspace(0, 1, 256)
img = np.outer(y, x).astype(np.float32)
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest -v tests/

# Run specific test file
pytest tests/test_gradient.py

# Run specific test
pytest tests/test_gradient.py::test_gradient_cross_backend

# Stop on first failure
pytest -x tests/

# Show print statements
pytest -s tests/
```
