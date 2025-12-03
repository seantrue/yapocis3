# Testing Basics

Testing kernels across backends ensures correctness and portability. This guide covers the essentials.

## Simple Test

```python
import numpy as np
import pytest
from yapocis.core.manager import get_backends
from yapocis.core import manager

@pytest.mark.parametrize("backend", get_backends())
def test_my_kernel(backend):
    # Create test data
    input_data = np.random.rand(100).astype(np.float32)
    expected = input_data * 2.0

    # Load and run kernel
    prog = manager.load_program("double", backend=backend)
    result = prog.double(input_data)

    # Verify
    np.testing.assert_allclose(result, expected, rtol=1e-5)
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_my_kernel.py -v

# Run for specific backend
pytest -m ocl      # OpenCL only
pytest -m metal    # Metal only
pytest -m numba    # Numba only
```

## Test Patterns

### Cross-Backend Comparison

```python
@pytest.mark.parametrize("backend", get_backends())
def test_cross_backend(backend):
    input_data = np.random.rand(256, 256).astype(np.float32)

    prog = manager.load_program("gradient", backend=backend)
    result = prog.gradient(input_data)

    # Compare against NumPy reference
    expected_dx = np.gradient(input_data, axis=1)
    expected_dy = np.gradient(input_data, axis=0)
    expected = np.sqrt(expected_dx**2 + expected_dy**2)

    np.testing.assert_allclose(result, expected, rtol=1e-4)
```

### Edge Cases

```python
def test_empty_input():
    prog = manager.load_program("double")
    input_data = np.array([], dtype=np.float32)
    result = prog.double(input_data)
    assert len(result) == 0

def test_single_element():
    prog = manager.load_program("double")
    input_data = np.array([0.5], dtype=np.float32)
    result = prog.double(input_data)
    assert result[0] == pytest.approx(1.0)

def test_large_array():
    prog = manager.load_program("double")
    input_data = np.random.rand(10000000).astype(np.float32)
    result = prog.double(input_data)
    np.testing.assert_allclose(result, input_data * 2.0, rtol=1e-5)
```

### Image Tests

```python
from yapocis.utils.basic import get_test_image

def test_with_image():
    img = get_test_image()

    prog = manager.load_program("blur")
    result = prog.blur(img, sigma=2.0)

    # Check output properties
    assert result.shape == img.shape
    assert result.dtype == np.float32
    assert 0 <= result.min() <= result.max() <= 1.0
```

## Validation Framework

Yapocis includes a validation framework for pixel-exact testing:

```python
@pytest.mark.validation
def test_graphics_exact(validate_result):
    canvas = np.zeros((512, 512, 4), dtype=np.float32)
    g = Graphics4()
    g.circles(canvas, [(256, 256, 100)], color=[1, 0, 0, 1])

    # Compare against stored reference
    validate_result("graphics_circles", g.read(canvas))
```

### Promoting Results

```bash
# View validation status
yapocis-validate info

# List pending validations
yapocis-validate list

# Promote a test result to reference
yapocis-validate promote "test_name"
```

## Debugging Tips

### Check Data Transfer

```python
def test_data_integrity():
    input_data = np.arange(100).astype(np.float32)

    # Use identity kernel to test data transfer
    prog = manager.load_program("identity")
    result = prog.identity(input_data)

    np.testing.assert_array_equal(result, input_data)
```

### Inspect Intermediate Values

```python
from yapocis.utils.basic import show_array

def test_debug():
    img = get_test_image()
    prog = manager.load_program("process")
    result = prog.process(img)

    # View result (saves to queue/raw/)
    show_array(result, "debug_output")
```

### Compare Backends

```python
def test_backend_parity():
    input_data = np.random.rand(256, 256).astype(np.float32)
    results = {}

    for backend in get_backends():
        prog = manager.load_program("gradient", backend=backend)
        results[backend] = prog.gradient(input_data)

    # All backends should produce same result
    reference = results[list(results.keys())[0]]
    for backend, result in results.items():
        np.testing.assert_allclose(
            result, reference, rtol=1e-4,
            err_msg=f"{backend} differs from reference"
        )
```

## Next Steps

- {doc}`/kernel-development/testing-kernels` - Advanced testing
- {doc}`/kernel-development/index` - Full kernel development guide
