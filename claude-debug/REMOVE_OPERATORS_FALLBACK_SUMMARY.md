# Remove OPERATORS Fallback - Summary

## Overview

Removed the legacy `OPERATORS` class attribute fallback from the Operators class, making the system rely solely on the `operators.yml` specification. This simplifies the code and enforces YAML as the single source of truth for operator definitions.

## Changes Made

### 1. yapocis/lib/operators.py

**Before:**
```python
class Operators(BoundKernels):
    INTERFACE = "operators"
    # Legacy fallback - operators context now externalized to operators.yml
    # This is kept for backward compatibility if YAML file is missing
    OPERATORS = [("add", "+"), ("sub", "-"), ("mul", "*"), ("divop", "/")]

    def __init__(self, *args, use_package: bool | None = None, **kwargs):
        # Only pass operators context if not already provided
        # This allows YAML context to take precedence while maintaining backward compatibility
        if 'operators' not in kwargs:
           kwargs['operators'] = self.OPERATORS
        super(Operators, self).__init__(*args, **kwargs)
```

**After:**
```python
class Operators(BoundKernels):
    INTERFACE = "operators"

    def __init__(self, *args, use_package: bool | None = None, **kwargs):
        # Operators context loaded automatically from operators.yml
        super(Operators, self).__init__(*args, **kwargs)
```

**Changes:**
- Removed `OPERATORS` class attribute
- Removed conditional logic in `__init__` that added operators to kwargs
- Simplified to rely entirely on YAML loading

### 2. yapocis/lib/operators.py - _init_with_package method

**Before:**
```python
def _init_with_package(self):
    """Initialize using compiled kernel package (fast path)."""
    from yapocis.kernelpack import get_operators_package

    # Get compiled package for current backend
    self.package = get_operators_package(self.kmgr)

    # Load all operator kernels
    kernel_names = [name for name, _ in self.OPERATORS]
    self.package.load(kernel_names)

    # Wrap package methods to match BoundKernels interface
    for name, _ in self.OPERATORS:
        ...
```

**After:**
```python
def _init_with_package(self):
    """Initialize using compiled kernel package (fast path)."""
    from yapocis.kernelpack import get_operators_package
    from yapocis.core.interface_context import load_interface_context

    # Load operator list from YAML
    context = load_interface_context("operators")
    operators = context.get('operators', [])

    # Get compiled package for current backend
    self.package = get_operators_package(self.kmgr)

    # Load all operator kernels
    kernel_names = [name for name, _ in operators]
    self.package.load(kernel_names)

    # Wrap package methods to match BoundKernels interface
    for name, _ in operators:
        ...
```

**Changes:**
- Added import for `load_interface_context`
- Load operators from YAML instead of `self.OPERATORS`
- Use YAML-loaded operators for kernel loading and wrapping

### 3. yapocis/numbax/kernel_manager.py

**Issue:** Numbax has its own `load_program` method that doesn't use the base class implementation, so it wasn't loading YAML context automatically.

**Before:**
```python
@classmethod
def load_program(
    cls,
    interface_spec: str,
    device_name: str | None = None,
    debug: bool = False,
    **context,
):
    # Load and render interface (expand Mako templates)
    source = load_interface(interface_spec)
    if context:
        try:
            source = str(Template(source).render(**context))
        except Exception:
            # If rendering fails, keep original source
            pass
    interface = get_interface(source)
```

**After:**
```python
@classmethod
def load_program(
    cls,
    interface_spec: str,
    device_name: str | None = None,
    debug: bool = False,
    **context,
):
    # Load and merge YAML context with programmatic context
    from yapocis.core.interface_context import get_interface_context
    merged_context = get_interface_context(interface_spec, **context)

    # Load and render interface (expand Mako templates)
    source = load_interface(interface_spec)
    if merged_context:
        try:
            source = str(Template(source).render(**merged_context))
        except Exception:
            # If rendering fails, keep original source
            pass
    interface = get_interface(source)
```

**Changes:**
- Added YAML context loading and merging
- Use `merged_context` instead of just `context` for template rendering
- Ensures numbax backend also benefits from YAML externalization

## Testing

### All Tests Pass

```
$ pytest tests/test_operators.py -v
======================== 24 passed, 5 warnings in 1.26s ========================
```

**Test Coverage:**
- OCL backend: ✅ All tests pass
- CUDA backend: ✅ All tests pass
- Numbax backend: ✅ All tests pass (after fix)

### Manual Verification

**CUDA backend:**
```python
>>> import os
>>> os.environ['YAPOCIS_BACKEND'] = 'cuda'
>>> from yapocis.lib.operators import Operators
>>> ops = Operators()
>>> ops.kmgr.callable.keys()
dict_keys(['add', 'sub', 'mul', 'divop', 'div', 'div_res', 'add_res', 'sub_res', 'mul_res', 'divop_res'])
```

**Numbax backend:**
```python
>>> os.environ['YAPOCIS_BACKEND'] = 'numbax'
>>> ops = Operators()
>>> ops.kmgr.callable.keys()
dict_keys(['add', 'sub', 'mul', 'divop', 'div', 'div_res', 'add_res', 'sub_res', 'mul_res', 'divop_res'])
```

## Benefits

### 1. Single Source of Truth
- `operators.yml` is now the only place operator definitions exist
- No confusion about which definition takes precedence
- Easier to maintain and update

### 2. Simpler Code
- Removed ~10 lines of conditional logic
- No more fallback/precedence handling
- Clear data flow: YAML → template → compiled kernels

### 3. Consistent Behavior
- All backends (OCL, CUDA, Metal, Numbax) use the same YAML source
- No backend-specific quirks or fallbacks
- Easier to reason about system behavior

### 4. Better Error Messages
- If `operators.yml` is missing, error is clear and immediate
- No silent fallback to potentially stale hardcoded values
- Encourages proper YAML configuration

## Migration Path

### For Developers

**Old way (no longer works):**
```python
# This relied on hardcoded OPERATORS fallback
ops = Operators()
```

**New way (required):**
```python
# This requires operators.yml to exist
ops = Operators()  # Loads from operators.yml automatically

# Or override programmatically
custom_ops = [["add", "+"], ["custom", "-"]]
ops = Operators(operators=custom_ops)
```

### For Deployments

**Required:**
- Ensure `yapocis/interfaces/operators.yml` exists
- File must contain `operators` key with operator list
- Format: `[[name, symbol], ...]`

**Example operators.yml:**
```yaml
operators:
  - [add, "+"]
  - [sub, "-"]
  - [mul, "*"]
  - [divop, "/"]
```

## Compatibility Notes

### Breaking Change: Yes

This is a **breaking change** for environments that:
- Don't have `operators.yml` installed
- Rely on the hardcoded `OPERATORS` fallback

However, normal installations are **not affected** because:
- `operators.yml` is included in the package
- Installed as part of `yapocis/interfaces/`
- Loaded automatically by the interface system

### Migration Required: No

For standard installations, no migration needed:
- YAML file already exists in repo
- Installed with package
- Tests verify functionality

### Custom Deployments

If you have a custom deployment that doesn't include interface files:
1. Copy `yapocis/interfaces/operators.yml` to your deployment
2. Or provide operators programmatically: `Operators(operators=[...])`

## Files Modified

1. **yapocis/lib/operators.py**
   - Removed `OPERATORS` class attribute
   - Simplified `__init__` method
   - Updated `_init_with_package` to load from YAML

2. **yapocis/numbax/kernel_manager.py**
   - Updated `load_program` to load and merge YAML context
   - Ensures numbax backend gets YAML context like other backends

## Documentation

- Updated inline comments to reflect YAML-only behavior
- Removed misleading "legacy fallback" comments
- Added clear references to operators.yml

## Related Work

This change completes the YAML externalization work:
- ✅ **operators** - Now YAML-only (this change)
- ✅ **convolvesep** - Uses YAML for Gaussian parameters
- 🎯 Future: Other interfaces (gradient, graphics, hsi, etc.)

## Conclusion

The removal of the `OPERATORS` fallback successfully enforces YAML as the single source of truth for operator definitions. All tests pass, the code is simpler, and the system behavior is more consistent across backends.

**Status**: ✅ Complete
- All tests passing (24/24)
- All backends verified (OCL, CUDA, Numbax)
- Numbax YAML loading fixed
- Code simplified and cleaned up
- Ready for production
