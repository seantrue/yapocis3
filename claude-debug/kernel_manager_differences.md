# Kernel Manager Differences: OCL vs Metal vs CUDA

## Summary

The three kernel managers (OpenCL, Metal, CUDA) share ~95% identical code structure and API. The differences are primarily in:
1. Template/kernel file locations
2. Backend-specific compilation APIs
3. Minor implementation details

## Structural Similarities

All three implement the same class structure:
```python
class KernelManager:
    def __init__(interface, device_name, debug, async_write, options, **context)
    def read(a)
    def write(a, async_write)
    def purge()
    def reset()
    def snapshot(**kwargs)
    def __getattr__(attr) -> Callable
    def stats()
    def __str__()
    @classmethod load_program(...)
    @classmethod import_program(...)
```

## Key Differences

### 1. Template/Kernel File Locations

**OpenCL (`yapocis/ocl/kernel_manager.py`)**
```python
from yapocis.kernels import find_kernel  # uses yapocis/kernels/*.mako

def render_program(base_filename, **context):
    tfilename = find_kernel(base_filename)
    mako_modules = os.path.join(get_yapocis_dir(), "mako_modules", "ocl")
```

**Metal (`yapocis/metallic/kernel_manager.py`)**
```python
def _find_kernel_template(base_filename):
    # Prefer Metal-specific templates
    base_dir = os.path.join(get_yapocis_dir(), "metalkernels")
    if os.path.exists(f"{base_dir}/{base_filename}.mako"):
        return that
    # Fallback to OCL kernels
    ocl_dir = os.path.join(get_yapocis_dir(), "kernels")
    return f"{ocl_dir}/{base_filename}.mako"

def render_program(base_filename, **context):
    tfilename = _find_kernel_template(base_filename)
    mako_modules = os.path.join(get_yapocis_dir(), "mako_modules", "metal")
```

**CUDA (`yapocis/cu/kernel_manager.py`)**
```python
from yapocis.cukernels import find_kernel  # uses yapocis/cukernels/*.mako

def render_program(base_filename, **context):
    tfilename = find_kernel(base_filename)
    mako_modules = os.path.join(get_yapocis_dir(), "mako_modules", "cuda")
```

### 2. Kernel Compilation

**OpenCL**
```python
import pyopencl

def __init__(...):
    self.mf = pyopencl.mem_flags
    self.program = self.dmgr.build(source)

    for kernel in interface.kernels:
        name = kernel.alias if kernel.alias else kernel.name
        self.callable[kernel.name] = Kernel(
            name, self,
            getattr(self.program, name),  # Direct attribute access
            kernel.parameters,
            async_write=async_write
        )
```

**Metal**
```python
import pymetallic as pm

def __init__(...):
    # No mem_flags needed
    # pymetallic compiles kernels on demand

    for kernel in interface.kernels:
        name = kernel.alias if getattr(kernel, "alias", None) else kernel.name
        pm_kernel = pm.Kernel(self.dmgr.device, source=source, func=name)
        self.callable[kernel.name] = Kernel(
            name, self, pm_kernel, kernel.parameters, async_write=async_write
        )
```

**CUDA**
```python
# No explicit import of pycuda here (in device_manager)

def __init__(...):
    # No mem_flags in CUDA manager

    # Save source to cache if YAPOCIS_CACHE is set
    cache_dir = os.environ.get("YAPOCIS_CACHE")
    if cache_dir and os.path.isdir(cache_dir):
        source_file = os.path.join(cache_dir, f"{interface.name}.cu")
        with open(source_file, "w") as f:
            f.write(source)

    self.program = self.dmgr.build(source, options=options)

    for kernel in interface.kernels:
        name = kernel.alias if kernel.alias else kernel.name
        kernel_func = self.program.get_function(name)  # Explicit get_function()
        self.callable[kernel.name] = Kernel(
            name, self, kernel_func, kernel.parameters, async_write=async_write
        )
```

### 3. Minor Implementation Details

**Metal-specific additions:**
- Imports `from __future__ import annotations`
- Has docstring at module level
- Includes `backend="metal"` in snapshot()
- Uses `device_name: str | int | None` type hints
- More defensive `getattr(kernel, "alias", None)`

**CUDA-specific additions:**
- Source code caching to `YAPOCIS_CACHE` directory
- Saves `.cu` files for debugging
- Updated copyright year to 2025
- `options` parameter passed through to build()

**OpenCL specifics:**
- Stores `self.mf = pyopencl.mem_flags` (used by Kernel class)
- Uses direct attribute access: `getattr(self.program, name)`

## Impact on Compiler Design

For a kernel compiler, the backend-specific differences are minimal:

1. **Import statements**: Change backend library imports
2. **Kernel retrieval**:
   - OCL: `getattr(program, name)`
   - Metal: `pm.Kernel(device, source, func=name)`
   - CUDA: `program.get_function(name)`
3. **Buffer flags**:
   - OCL: Uses `cl.mem_flags`
   - Metal: PyMetallic handles internally
   - CUDA: PyCUDA handles internally

The compiled module would need backend-specific sections for these three areas, but the overall parameter handling logic (handle_in, handle_out_like, etc.) is **identical** across all backends.

## Recommendation

A kernel compiler should:
1. Generate a **common base class** with all parameter handling logic
2. Generate **backend-specific subclasses** that only override:
   - Kernel retrieval/compilation
   - Buffer allocation (mem_flags handling)
   - Import statements

This maximizes code reuse and minimizes duplication.
