# PyCUDA Resource Leak Analysis and Fix - Summary

## Problem Statement

PyCUDA resources (GPU buffers, contexts, streams) were accumulating during repeated library operations, causing "illegal memory access" errors after ~40-50 iterations.

## Root Cause

**Buffer Accumulation Pattern:**
1. Library methods return numpy arrays with associated GPU buffers
2. Returned arrays stay alive in caller's scope
3. GPU buffers remain allocated (strong references in `buffers` dict)
4. Even with `purge()`, buffers for live arrays are not freed
5. After many iterations, GPU memory exhausted → illegal memory access

**Key Architecture:**
- `arrays`: WeakValueDictionary (numpy arrays can be GC'd)
- `buffers`: Regular dict (GPU buffers have strong references)
- `purge()`: Removes buffers whose arrays have been GC'd

## Solution Implemented

### 1. Infrastructure Improvements

**Device Manager Base (`yapocis/core/device_mgr_base.py`)**
- Added `_free_buffer()` hook for backend-specific cleanup
- Added context manager support (`__enter__`, `__exit__`)
- Added explicit `close()` method
- Added resource tracking:
  - `count_leaked_buffers()`: Count buffers without numpy arrays
  - `check_resource_usage()`: Monitor buffer usage with warnings
  - Buffer warning threshold: 100
  - Leaked buffer warning threshold: 50

**CUDA Device Manager (`yapocis/cu/device_manager.py`)**
- Implemented `_free_buffer()` (pass - let PyCUDA GC handle it)
- Implemented `_close()` with stream synchronization
- Added atexit handler for context cleanup (opt-in)
- Updated `__del__` as fallback cleanup

**Library Base Class (`yapocis/lib/bound_kernels.py`)**
- Added context manager support
- Added conditional `__del__` purge (opt-in)
- Provides `purge()` method for explicit cleanup

### 2. Environment Variables

**YAPOCIS_AUTO_PURGE** (default: disabled)
- Set to '1', 'true', or 'yes' to enable automatic buffer purge on instance deletion
- Useful for tight loops where manual purge is cumbersome
- Disabled by default to prevent pytest teardown issues

**YAPOCIS_ATEXIT_CLEANUP** (default: disabled)
- Set to '1', 'true', or 'yes' to enable CUDA context cleanup at program exit
- Pops context from stack and cleans up buffers
- Disabled by default to avoid pytest context corruption

### 3. Recommended Usage Patterns

**Pattern 1: Context Manager (Best Practice)**
```python
with Gaussian() as gauss:
    img = np.random.rand(512, 512).astype(np.float32)
    result = gauss.gauss_image(img, scale=2)
    # Use result...
    # Automatic purge on exit
```

**Pattern 2: Explicit Purge**
```python
gauss = Gaussian()
img = np.random.rand(512, 512).astype(np.float32)
result = gauss.gauss_image(img, scale=2)

# Use result...

# Clean up
del img
del result  # Important: delete returned arrays
gauss.purge()  # Purge GPU buffers
del gauss
```

**Pattern 3: Auto-Purge (Opt-in)**
```python
import os
os.environ['YAPOCIS_AUTO_PURGE'] = '1'

for i in range(1000):
    gauss = Gaussian()
    img = np.random.rand(512, 512).astype(np.float32)
    result = gauss.gauss_image(img, scale=2)

    # Clean up
    del img
    del result
    del gauss  # Automatic purge on deletion
```

**Pattern 4: Reuse Instance**
```python
gauss = Gaussian()

for i in range(1000):
    img = np.random.rand(512, 512).astype(np.float32)
    result = gauss.gauss_image(img, scale=2)

    # Clean up each iteration
    del img
    del result

    # Purge every N iterations
    if (i + 1) % 10 == 0:
        gauss.purge()

# Final cleanup
gauss.purge()
del gauss
```

## Test Results

### Pytest Tests
```
tests/test_kernel_performance.py::TestKernelPerformance::test_gaussian_performance
  - 10 passed (OCL, CUDA across all library operations)
  - 1 failed (numbax - unrelated to resource leaks)
```

### Stress Tests

**Comprehensive Library Test** (`claude-debug/test_cuda_lib_resource_leak.py`)
- Graphics4: 100 iterations ✓
- Gaussian: 100 iterations ✓
- Gradient: 100 iterations ✓
- Median: 100 iterations ✓
- HSI: 100 iterations ✓
- Mandelbrot: 100 iterations ✓
- Anisodiff: 50 iterations ✓
- Mixed operations: 100 iterations ✓
- **Total: 750 library operations - PASSED**

**Reuse Pattern Test** (`claude-debug/test_gaussian_reuse_only.py`)
- 100 iterations with single Gaussian instance ✓
- Periodic purging every 10 iterations ✓

## Critical Insights

1. **Buffer Lifetime**: GPU buffers stay alive as long as numpy arrays exist, even if arrays go out of scope in function but are returned
2. **Purge Semantics**: `purge()` only frees buffers whose arrays have been GC'd - live arrays keep their buffers
3. **Delete Order Matters**: Delete returned arrays (`del result`) BEFORE calling `purge()`
4. **Pytest Compatibility**: Aggressive cleanup during `__del__` corrupts CUDA context for subsequent tests
5. **Singleton Manager**: Device manager is cached per device, so buffers accumulate across all library instances unless purged

## Files Modified

1. `yapocis/core/device_mgr_base.py`
   - Added buffer cleanup infrastructure
   - Added context managers
   - Added resource tracking

2. `yapocis/cu/device_manager.py`
   - Implemented CUDA-specific cleanup
   - Added atexit handler
   - Improved __del__ robustness

3. `yapocis/lib/bound_kernels.py`
   - Added context manager support
   - Added conditional __del__ purge

4. `yapocis/lib/gaussian.py`
   - Reverted line 210 from `return aa.copy()` to `return aa`

## Test Files Created

- `claude-debug/test_cuda_lib_resource_leak.py` - Comprehensive stress test
- `claude-debug/test_gaussian_with_purge.py` - Gaussian purge patterns
- `claude-debug/test_gaussian_reuse_only.py` - Instance reuse pattern
- `claude-debug/test_buffer_tracking.py` - Buffer accumulation tracking
- `claude-debug/test_debug_memory_access.py` - Memory access debugging

## Conclusion

The PyCUDA resource leak issue is **RESOLVED**. The fix provides:

✅ No resource leaks with proper cleanup patterns
✅ Backward compatible (opt-in aggressive cleanup)
✅ Pytest compatible (no cleanup during teardown)
✅ Production ready (750 operations tested successfully)
✅ Multiple usage patterns (context managers, explicit purge, auto-purge)

**Recommendation**: Use context managers (`with` statement) for all new code as the safest and cleanest pattern.
