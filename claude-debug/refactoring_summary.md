# Numbax Kernel Refactoring Summary

## Goal

Eliminate the runtime interpretation overhead in numbax kernels by providing direct callable classes instead of wrapping functions in the `Kernel` interpreter.

## Test Results

✅ **Proof of concept works!**
- NumPy implementation matches old implementation perfectly (0 pixel difference)
- JIT implementation has minor differences (±1 iteration, acceptable)
- Code is much simpler and more direct

## What Needs to Change

### 1. Kernel Manager (`yapocis/numbax/kernel_manager.py`)

**Current**: 188 lines with complex `_make_dynamic_fn` logic
**New**: ~150 lines, much simpler

Key changes:
- Remove `Kernel` import and wrapper usage
- Add `_to_class_name()` helper (converts `mandelbrot` → `Mandelbrot`)
- Change kernel lookup from `getattr(mod, func_name)` to `getattr(mod, class_name + 'JIT')`
- Instantiate classes: `self.callable[k.name] = cls()`
- Add `use_jit` parameter to choose NumPy vs JIT variants

### 2. Remove Kernel Wrapper (`yapocis/numbax/kernel.py`)

**Current**: 90 lines of parameter interpretation
**After**: DELETE THIS FILE (no longer needed!)

This file contains all the complexity we're eliminating:
- Parameter mapping (`posbyname` dictionary)
- Dimension derivation (`width_of`, `height_of`, etc.)
- Output allocation (`out_like` hint handling)
- Return value construction

### 3. Refactor All Kernel Modules

Each kernel module in `yapocis/numbax/kernels/*.py` needs:

**Pattern for simple kernels** (mandelbrot, imageops, colormath, etc.):
```python
# OLD:
def kernel_name(input, output, param1, param2):
    # implementation
    return output

# NEW:
class KernelName:
    """NumPy implementation"""
    def __call__(self, input, param1, param2):
        output = np.zeros(input.shape, dtype=...)
        # implementation
        return output

class KernelNameJIT:
    """Numba JIT implementation"""
    def __call__(self, input, param1, param2):
        output = np.zeros(input.shape, dtype=...)
        _kernel_name_njit(input, output, param1, param2)
        return output
```

**Pattern for complex kernels** (graphics4, convolvesep):
```python
# OLD:
def lines(width, height, a, nelements, coords, widths, colors):
    # implementation
    return a

# NEW:
class Lines:
    def __call__(self, a, coords, widths, colors):
        # Derive dimensions from inputs
        height, width = a.shape[:2]
        nelements = len(coords)
        # implementation
        return a

class LinesJIT:
    def __call__(self, a, coords, widths, colors):
        # Same but with JIT helpers
        return a
```

## Files to Modify

### Core (2 files)
- ✏️ `yapocis/numbax/kernel_manager.py` - Simplify to ~150 lines
- ❌ `yapocis/numbax/kernel.py` - DELETE (90 lines eliminated!)

### Kernels (16 files) - All in `yapocis/numbax/kernels/`
1. ✏️ `mandelbrot.py` - Simple (1 kernel)
2. ✏️ `imageops.py` - Simple (4 kernels: clamp_pixels, stretch, binarize, histogram)
3. ✏️ `colormath.py` - Simple (color space conversions)
4. ✏️ `hsi.py` - Simple (HSI transforms)
5. ✏️ `gradient.py` - Simple (gradient computation)
6. ✏️ `median3x3.py` - Simple (median filter)
7. ✏️ `zcs.py` - Simple (zero-crossing detection)
8. ✏️ `demo.py` - Simple (demo kernels)
9. ✏️ `anisodiff.py` - Medium complexity
10. ✏️ `graphics.py` - Medium complexity (multiple primitives)
11. ✏️ `graphics4.py` - Medium complexity (6 kernels: lines, rects, circles, triangles, ellipses, composite)
12. ✏️ `truetrace.py` - Complex (SDF path tracing)
13. ✏️ `convolve.py` - Complex (parameterized interface with `_fast` variants)
14. ✏️ `convolvesep.py` - Complex (parameterized with `_h`, `_v` variants)
15. ✏️ `operators.py` - Special case (may need custom handling)

### Tests
- All existing tests should pass unchanged (API is backward compatible)
- May want to add specific tests for new class-based kernels

## Migration Strategy

### Phase 1: Core + Simple Kernels (Prove It Works)
1. Update `kernel_manager.py` to support BOTH old and new style
2. Delete `kernel.py` Kernel wrapper
3. Migrate 3-4 simple kernels (mandelbrot, imageops, median3x3)
4. Run full test suite to verify

### Phase 2: Medium Complexity
5. Migrate graphics, graphics4, anisodiff
6. Test graphics operations thoroughly

### Phase 3: Complex Kernels
7. Migrate convolve, convolvesep (handle parameterized interfaces)
8. Migrate truetrace
9. Handle operators special cases

### Phase 4: Cleanup
10. Remove any old-style fallback code
11. Update documentation
12. Performance testing

## Benefits

### Code Quality
- **90 lines eliminated** (Kernel wrapper deleted)
- **Simpler kernel manager** (~38 lines simpler)
- **Clearer kernel code** (classes with explicit signatures)
- **Better type hints** (direct method signatures)

### Performance
- **Zero interpretation overhead** (direct function calls)
- **No parameter mapping** (no posbyname dictionary)
- **No dimension derivation loop** (computed locally if needed)
- **No output allocation overhead** (done once in __call__)

### Developer Experience
- **Easier to write** (just implement __call__)
- **Easier to test** (instantiate class directly)
- **Easier to debug** (no wrapper layer)
- **Clear separation** (NumPy vs JIT explicit)

## Risks

### Low Risk
- ✅ Backward compatible (API unchanged for users)
- ✅ Test suite verifies correctness
- ✅ Can be done incrementally (kernel by kernel)

### Medium Risk
- ⚠️ Parameterized interfaces (convolve, convolvesep) need careful handling
- ⚠️ Extension kernels (_res, _fast, _h, _v) need naming conventions
- ⚠️ Some tests may depend on internal Kernel class details

## Next Steps

Ready to proceed with Phase 1?

1. Update kernel_manager.py
2. Migrate mandelbrot.py (DONE - tested in claude-debug/)
3. Migrate imageops.py
4. Run tests to verify
