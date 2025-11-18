# Backend Standardization

## Overview

Standardized all parametrized backend tests to use `get_test_backends()` instead of `get_backends()`. This enables consistent backend filtering across the entire test suite using the `YAPOCIS_TEST_BACKENDS` environment variable.

## Changes Made

### Files Updated (7 files)

All parametrized tests that used `get_backends()` have been updated to use `get_test_backends()`:

1. `tests/backend/test_smoke.py`
2. `tests/lib/compute/test_operators.py`
3. `tests/lib/fft/test_fft.py`
4. `tests/lib/fft/test_fft_library.py`
5. `tests/lib/graphics/test_antialiasing.py`
6. `tests/performance/test_library_perf.py`
7. `tests/regression/test_validation.py`

### What Changed

**Before:**
```python
from yapocis.core import get_backends

@pytest.mark.parametrize("backend", get_backends())
def test_operation(backend):
    ...
```

**After:**
```python
from yapocis.core import get_backends, get_test_backends

@pytest.mark.parametrize("backend", get_test_backends())
def test_operation(backend):
    ...
```

## Benefits

### 1. Consistent Backend Filtering

You can now filter backends consistently across all tests using the `YAPOCIS_TEST_BACKENDS` environment variable:

```bash
# Run tests on only OCL backend
YAPOCIS_TEST_BACKENDS=ocl pytest tests/

# Run tests on OCL and numbax backends
YAPOCIS_TEST_BACKENDS=ocl,numbax pytest tests/

# Run graphics tests on metal only
YAPOCIS_TEST_BACKENDS=metal pytest tests/lib/graphics/
```

### 2. Faster Development Cycles

When working on a specific backend, you can limit test runs to just that backend:

```bash
# Only test numbax backend during numba development
YAPOCIS_TEST_BACKENDS=numbax pytest tests/
```

### 3. CI/CD Flexibility

Different CI/CD jobs can test different backends in parallel:

```yaml
test-ocl:
  env:
    YAPOCIS_TEST_BACKENDS: ocl
  run: pytest tests/

test-metal:
  env:
    YAPOCIS_TEST_BACKENDS: metal
  run: pytest tests/

test-numbax:
  env:
    YAPOCIS_TEST_BACKENDS: numbax
  run: pytest tests/
```

### 4. Smoke Test Optimization

Run quick smoke tests on a single backend:

```bash
# Fast smoke test on OCL only
YAPOCIS_TEST_BACKENDS=ocl pytest -m smoke tests/
```

## How `get_test_backends()` Works

The function respects backend filtering in this priority order:

1. **Explicit parameter**: `get_test_backends(['ocl', 'metal'])`
2. **Environment variable**: `YAPOCIS_TEST_BACKENDS=ocl,metal`
3. **All available backends**: Returns all detected backends (default)

## Verification

### Test Collection Verified

- ✅ **All tests**: 655 tests collected
- ✅ **Smoke tests**: 89 tests collected
- ✅ **Backend filtering**: Working correctly

### Backend Filtering Example

Using `tests/lib/compute/test_operators.py` as an example:

| Filter | Tests Collected | Explanation |
|--------|----------------|-------------|
| None (all backends) | 24 tests | 7 parametrized × 3 backends + 3 non-parametrized |
| `ocl` only | 10 tests | 7 parametrized × 1 backend + 3 non-parametrized |
| `ocl,numbax` | 17 tests | 7 parametrized × 2 backends + 3 non-parametrized |

## Usage Examples

### Development Workflow

```bash
# Test only the backend you're working on
YAPOCIS_TEST_BACKENDS=metal pytest tests/lib/graphics/

# Quick smoke test on one backend
YAPOCIS_TEST_BACKENDS=ocl pytest -m smoke tests/

# Test specific module on two backends
YAPOCIS_TEST_BACKENDS=ocl,numbax pytest tests/lib/fft/
```

### Correctness Testing

```bash
# Run all graphics tests on OCL and numbax only
YAPOCIS_TEST_BACKENDS=ocl,numbax pytest tests/lib/graphics/

# Run all correctness tests (exclude performance) on metal
YAPOCIS_TEST_BACKENDS=metal pytest -m "not performance" tests/
```

### Performance Testing

```bash
# Performance benchmarks on specific backend
YAPOCIS_TEST_BACKENDS=metal pytest -m performance tests/

# Compare performance across two backends
YAPOCIS_TEST_BACKENDS=ocl,metal pytest -m performance tests/performance/
```

## Notes

- All 655 tests remain functional after standardization
- No breaking changes to test behavior
- Tests without `YAPOCIS_TEST_BACKENDS` set will test all available backends (default behavior)
- Backend-specific markers (`@pytest.mark.ocl`, etc.) are still respected

## Date

November 6, 2025
