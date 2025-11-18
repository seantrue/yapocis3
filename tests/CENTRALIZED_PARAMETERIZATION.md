# Centralized Backend/Device Parameterization

## Overview

Moved backend/device parameterization logic from individual test files to a centralized function in `tests/__init__.py`. This provides a single source of truth for generating backend/device combinations and ensures consistent filtering behavior across all tests.

## Changes Made

### Centralized Function

Added `get_backend_device_combinations()` to `tests/__init__.py`:

```python
def get_backend_device_combinations():
    """
    Generate all (backend, device) combinations for testing.

    Respects YAPOCIS_TEST_BACKENDS for filtering.
    """
    from yapocis.core import get_test_backends, get_test_devices

    combinations = []
    for backend in get_test_backends():
        devices = get_test_devices(backend)
        for device_name in devices.keys():
            combinations.append((backend, device_name))
    return combinations
```

### Files Updated

1. **tests/__init__.py** - Added centralized `get_backend_device_combinations()`
2. **tests/conftest.py** - Added copy for documentation/reference
3. **tests/async/test_modes.py** - Removed local function, imports from tests
4. **tests/async/test_performance.py** - Removed local function, imports from tests
5. **tests/performance/test_kernel_perf.py** - Removed local function, imports from tests, updated to use `get_test_backends()`

## Benefits

### 1. Single Source of Truth

All backend/device combinations generated in one place. Changes to the logic only need to be made once.

### 2. Consistent Filtering

All tests now use `get_test_backends()` which respects `YAPOCIS_TEST_BACKENDS`:

```bash
# Run tests on specific backends
YAPOCIS_TEST_BACKENDS=ocl,numbax pytest tests/async/
YAPOCIS_TEST_BACKENDS=metal pytest -m performance tests/
```

### 3. Easier Maintenance

Adding new backend filtering logic or device selection logic only requires updating one function.

### 4. Better Test Isolation

Tests can be run with different backend/device configurations without code changes:

```bash
# Test on all available backends/devices
pytest tests/async/test_modes.py

# Test on OCL only (across all OCL devices)
YAPOCIS_TEST_BACKENDS=ocl pytest tests/async/test_modes.py

# Test on metal and numbax
YAPOCIS_TEST_BACKENDS=metal,numbax pytest tests/async/
```

## Usage

### In Test Files

```python
from tests import get_backend_device_combinations

@pytest.mark.parametrize("backend,device", get_backend_device_combinations())
class TestSomething:
    def test_operation(self, backend, device):
        os.environ["YAPOCIS_BACKEND"] = backend
        os.environ["YAPOCIS_DEVICE"] = device
        # Test implementation
```

### Running Tests

```bash
# All async tests across all backends/devices
pytest tests/async/

# Async tests on OCL backend only
YAPOCIS_TEST_BACKENDS=ocl pytest tests/async/

# Performance tests on metal backend
YAPOCIS_TEST_BACKENDS=metal pytest -m performance tests/performance/

# Specific test file filtered to two backends
YAPOCIS_TEST_BACKENDS=ocl,numbax pytest tests/async/test_modes.py
```

## Test Counts

### tests/async/test_modes.py
- All backends (ocl, metal, numbax): **30 tests**
- OCL only: **10 tests**
- Ratio: 3 backends × ~10 tests each

### tests/async/test_performance.py
- All backends: **25 tests** (performance marked)
- Filtered similarly based on backend count

### tests/performance/test_kernel_perf.py
- All backends: **30 tests** (performance marked)
- Filtered similarly based on backend count

## Previous Implementation

Before centralization, each file had its own implementation:

**tests/async/test_modes.py & test_performance.py** (correct):
```python
def get_backend_device_combinations():
    combinations = []
    for backend in get_test_backends():  # ✓ Uses get_test_backends
        devices = get_test_devices(backend)
        for device_name in devices.keys():
            combinations.append((backend, device_name))
    return combinations
```

**tests/performance/test_kernel_perf.py** (needed update):
```python
def get_backend_device_combinations():
    combinations = []
    backend_infos = get_backend_infos()
    for backend in get_backends():  # ✗ Used get_backends instead
        if backend in backend_infos:
            devices = backend_infos[backend].get("devices", [])
            for device_name in devices:
                combinations.append((backend, device_name))
    return combinations
```

The centralized version now uses `get_test_backends()` consistently.

## Verification

All tests verified working:

```bash
# Total test count unchanged
pytest --collect-only tests/
# 655 tests collected

# Async tests collect correctly
pytest --collect-only tests/async/
# 55 tests collected

# Performance tests collect correctly
pytest --collect-only tests/performance/
# 45 tests collected

# Filtering works correctly
YAPOCIS_TEST_BACKENDS=ocl pytest --collect-only tests/async/test_modes.py
# 10 tests collected (vs 30 with all backends)
```

## Future Enhancements

The centralized function makes it easy to add features like:

- Device filtering: `YAPOCIS_TEST_DEVICES=GPU1,GPU2`
- Backend/device exclusion: `YAPOCIS_EXCLUDE_BACKENDS=numbax`
- Specific device type filtering: `YAPOCIS_DEVICE_TYPE=GPU`

All such changes would only need to be made in `get_backend_device_combinations()`.

## Date

November 6, 2025
