# NumPy Backend Design

## Goal

Create a separate `numpy` backend that uses pure NumPy implementations (no Numba JIT), selectable via `YAPOCIS_BACKEND=numpy`.

## Architecture

### Backend Separation

**Current situation:**
- `numbax` backend uses `Mandelbrot` or `MandelbrotJIT` classes based on `use_jit` parameter
- Single backend with two code paths

**New situation:**
- `numpy` backend: Pure NumPy, always uses `Mandelbrot` class
- `numbax` backend: Numba JIT, always uses `MandelbrotJIT` class (or keeps current behavior)
- Clean separation, different validation thresholds

### Directory Structure

```
yapocis/
├── numpy/                      # NEW: Pure NumPy backend
│   ├── __init__.py
│   ├── kernel_manager.py       # Simplified KernelManager
│   └── kernels/
│       ├── __init__.py
│       ├── mandelbrot.py       # Class: Mandelbrot (pure NumPy)
│       ├── imageops.py         # Classes: ClampPixels, Stretch, etc.
│       ├── graphics4.py        # Classes: Lines, Circles, etc.
│       └── ...                 # All other kernels
│
├── numbax/                     # EXISTING: Numba JIT backend
│   ├── __init__.py
│   ├── kernel_manager.py       # Either keep current or switch to JIT-only
│   └── kernels/
│       ├── mandelbrot.py       # Could keep both or just JIT
│       └── ...
│
└── core/
    └── manager.py              # Add 'numpy' to get_backends()
```

### Implementation Strategy

**Option 1: Share kernel code (symlinks/imports)**
- `numpy/kernels/` imports from shared location
- `numbax/kernels/` imports from shared location
- Reduces duplication but adds indirection

**Option 2: Copy kernel implementations**
- `numpy/kernels/` has pure NumPy classes
- `numbax/kernels/` has JIT classes
- More duplication but cleaner separation

**Recommended: Option 2** - Clean separation, easier to maintain

### Kernel Manager Differences

**numpy/kernel_manager.py:**
```python
class KernelManager:
    def __init__(self, interface, **kwargs):
        mod = importlib.import_module(f"yapocis.numpy.kernels.{interface.name}")
        for k in interface.kernels:
            class_name = _to_class_name(k.name)
            cls = getattr(mod, class_name)  # Only NumPy class
            self.callable[k.name] = cls()
```

**numbax/kernel_manager.py (option A - JIT only):**
```python
class KernelManager:
    def __init__(self, interface, **kwargs):
        mod = importlib.import_module(f"yapocis.numbax.kernels.{interface.name}")
        for k in interface.kernels:
            class_name = _to_class_name(k.name) + "JIT"
            cls = getattr(mod, class_name)  # Only JIT class
            self.callable[k.name] = cls()
```

**numbax/kernel_manager.py (option B - keep both, default JIT):**
- Keep current behavior with `use_jit=True` default
- Allows `YAPOCIS_BACKEND=numbax` with `use_jit=False` for testing

**Recommended: Option A** - Numbax means JIT, numpy means pure NumPy

## Migration Path

### Phase 1: Create numpy backend structure
1. Create `yapocis/numpy/` directory
2. Create `yapocis/numpy/__init__.py`
3. Create `yapocis/numpy/kernels/` directory

### Phase 2: Implement numpy kernel manager
1. Copy `numbax/kernel_manager.py` to `numpy/kernel_manager.py`
2. Simplify to only load NumPy classes (no JIT)
3. Remove `_make_dynamic_fn` complexity (start simple)

### Phase 3: Port simple kernel
1. Create `numpy/kernels/mandelbrot.py` with `Mandelbrot` class
2. Test with `YAPOCIS_BACKEND=numpy`

### Phase 4: Update core manager
1. Add `'numpy'` to `get_backends()` in `core/manager.py`
2. Map `'numpy'` → `yapocis.numpy.kernel_manager`

### Phase 5: Decide on numbax backend
- Keep current behavior (both NumPy and JIT)?
- Or make it JIT-only?

### Phase 6: Port remaining kernels
- Port all kernels from `numbax/kernels/` to `numpy/kernels/`
- Extract pure NumPy implementations

## Benefits

### Clean Separation
- `numpy`: Pure NumPy, no Numba dependency
- `numbax`: Fast Numba JIT
- `ocl`, `metal`, `cuda`: GPU backends

### Different Validation Thresholds
```python
# In tests
if backend == "numpy":
    assert np.array_equal(result, expected)  # Perfect match
elif backend == "numbax":
    assert np.allclose(result, expected, atol=1)  # Allow ±1 difference
elif backend in ["ocl", "metal", "cuda"]:
    assert np.allclose(result, expected, rtol=1e-5)  # Float tolerance
```

### Easier Testing
- `YAPOCIS_BACKEND=numpy` for reference results
- `YAPOCIS_BACKEND=numbax` for performance
- No need to toggle `use_jit` parameter

### Clearer Intent
- numpy = reference implementation (correctness)
- numbax = fast implementation (performance)
- GPU backends = very fast (max performance)

## Risks

### Code Duplication
- Will have two copies of each kernel
- Mitigated by: kernels are relatively small, and implementations diverge anyway

### Maintenance
- Need to update both `numpy/` and `numbax/` kernels
- Mitigated by: test suite catches regressions

### Test Coverage
- Need to ensure both backends are tested
- Already have `@pytest.mark.parametrize("backend", get_backends())`

## Next Steps

1. Create directory structure
2. Implement minimal kernel manager
3. Port mandelbrot kernel
4. Update core manager
5. Run test suite
