# Kernel Packager - Revised Design (Gradual Migration)

## Goals

1. **Zero Breaking Changes**: Existing code continues to work unchanged
2. **Gradual Migration**: Opt-in on a per-class basis
3. **Performance**: Eliminate runtime interpretation overhead
4. **Target**: `Gaussian` class using `convolvesep` interface

## Directory Structure

```
yapocis/
├── kernelpack/           # NEW: Generated kernel packages
│   ├── __init__.py
│   ├── ocl/             # OpenCL packages
│   │   ├── __init__.py
│   │   └── convolvesep.py
│   ├── metal/           # Metal packages
│   │   ├── __init__.py
│   │   └── convolvesep.py
│   ├── cuda/            # CUDA packages
│   │   ├── __init__.py
│   │   └── convolvesep.py
│   └── numba/           # Numba packages
│       ├── __init__.py
│       └── convolvesep.py
├── lib/
│   ├── gaussian.py      # MODIFIED: Optionally use kernel package
│   └── bound_kernels.py # UNCHANGED: Still works
└── core/
    └── ...              # UNCHANGED
```

## Key Pattern: The Gaussian Convolution Interface

### Current Interface (Mako template with extensions)

```idl
interface convolvesep {
%for name,_ in convs:
    kernel ${name}(int vertical, width_of input, height_of input, in float input, out_like input);
    extension res of ${name}(int vertical, width_of input, height_of input, resident float input, resident float output);
    kernel ${name}_h(width_of input, height_of input, resident float input, resident float output);
    kernel ${name}_v(width_of input, height_of input, resident float input, resident float output);
%endfor
};
```

This generates kernels like:
- `gauss3(vertical, ...)` - Main kernel with copy
- `gauss3_res(vertical, ...)` - Extension using resident buffers
- `gauss3_h(...)` - Horizontal pass only (resident)
- `gauss3_v(...)` - Vertical pass only (resident)

### Current Usage Pattern in Gaussian

```python
class Gaussian(BoundKernels):
    def __init__(self, device_name=None, gaussian_weights=None):
        gws = self.gaussian_weights.gaussian_weights()
        convs = [("gauss%s" % len(a), a) for a in gws]
        super().__init__(device_name=device_name, convs=convs)

        # Get kernel references
        kmgr = self.kmgr
        self.gaussian_kernels = [getattr(kmgr, name) for (name, conv) in convs]

        # Attach extension methods to each kernel
        for i, (name, conv) in enumerate(convs):
            self.gaussian_kernels[i].res = getattr(kmgr, f"{name}_res")
            self.gaussian_kernels[i].h = getattr(kmgr, f"{name}_h")
            self.gaussian_kernels[i].v = getattr(kmgr, f"{name}_v")
            self.gaussian_kernels[i].width = len(conv)

    def gauss_image(self, a, scale):
        g = self.get_gaussian_kernel(scale)
        aa = a.copy()
        b = np.empty_like(aa)
        g.write(aa)
        g.write(b)
        g.h(aa, b)      # Horizontal pass
        g.v(b, aa)      # Vertical pass
        g.read(aa)
        return aa
```

## Generated Package Structure

### Example: `yapocis/kernelpack/ocl/convolvesep.py`

```python
"""
Auto-generated OpenCL kernel package for convolvesep interface.
Generated from: yapocis/interfaces/convolvesep.interface
Backend: OpenCL
DO NOT EDIT - Regenerate with: yapocis-package convolvesep --backend ocl
"""

import numpy as np
from typing import Optional
from yapocis.ocl.device_manager import DeviceManager


class ConvolvesepPackage:
    """
    Compiled kernel package for separable convolution operations.

    Provides zero-overhead calls to convolution kernels by pre-compiling
    all buffer management logic.
    """

    def __init__(self, dmgr: DeviceManager, program):
        """
        Initialize package with device manager and compiled program.

        Args:
            dmgr: OpenCL device manager
            program: Compiled pyopencl.Program
        """
        self.dmgr = dmgr
        self.queue = dmgr.queue
        self.autoflatten = dmgr.autoflatten

        # Store kernel references (populated by load())
        self._kernels: dict[str, any] = {}

    def load(self, kernel_names: list[str]) -> None:
        """
        Load specific kernels from the program.

        Args:
            kernel_names: List of kernel names to load (e.g., ['gauss3', 'gauss5'])
        """
        for name in kernel_names:
            # Main kernel
            self._kernels[name] = getattr(self.dmgr.program, name)
            # Extensions
            self._kernels[f"{name}_res"] = getattr(self.dmgr.program, f"{name}_res")
            self._kernels[f"{name}_h"] = getattr(self.dmgr.program, f"{name}_h")
            self._kernels[f"{name}_v"] = getattr(self.dmgr.program, f"{name}_v")

    def gauss(
        self,
        kernel_name: str,
        vertical: int,
        input_array: np.ndarray
    ) -> np.ndarray:
        """
        Execute separable convolution kernel.

        Interface:
            kernel ${name}(int vertical, width_of input, height_of input,
                          in float input, out_like input);

        Args:
            kernel_name: Name of kernel (e.g., 'gauss3', 'gauss5')
            vertical: 0 for horizontal, 1 for vertical
            input_array: Input image (float32)

        Returns:
            Output array with same shape as input
        """
        # Param 0: int vertical (scalar)
        vertical_i32 = np.int32(vertical)

        # Param 1: width_of input (derived)
        width = np.int32(input_array.shape[0])

        # Param 2: height_of input (derived)
        height = np.int32(input_array.shape[1])

        # Param 3: in float input
        input_array = input_array.astype(np.float32, copy=False)
        buf_input = self.dmgr.write(input_array, wait=True)

        # Param 4: out_like input
        output = np.empty_like(input_array, dtype=np.float32)
        buf_output = self.dmgr.ensure(output)

        # Execute kernel
        global_size = (input_array.size,) if self.autoflatten else input_array.shape
        kernel = self._kernels[kernel_name]
        kernel(
            self.queue,
            global_size,
            None,  # local_size
            vertical_i32,
            width,
            height,
            buf_input,
            buf_output
        )

        # Read result
        self.dmgr.read(output)
        return output

    def gauss_res(
        self,
        kernel_name: str,
        vertical: int,
        input_array: np.ndarray,
        output_array: np.ndarray
    ) -> None:
        """
        Execute separable convolution with resident buffers.

        Interface:
            extension res of ${name}(int vertical, width_of input, height_of input,
                                    resident float input, resident float output);

        Args:
            kernel_name: Name of kernel (e.g., 'gauss3', 'gauss5')
            vertical: 0 for horizontal, 1 for vertical
            input_array: Input image already on GPU
            output_array: Output buffer already on GPU

        Note:
            Arrays must already be resident on GPU (via write() or ensure())
            No automatic read() - caller must explicitly read results
        """
        # Param 0: int vertical (scalar)
        vertical_i32 = np.int32(vertical)

        # Param 1: width_of input (derived)
        width = np.int32(input_array.shape[0])

        # Param 2: height_of input (derived)
        height = np.int32(input_array.shape[1])

        # Param 3: resident float input (already on GPU)
        buf_input = self.dmgr.buffers.get(id(input_array))
        if buf_input is None:
            raise ValueError(f"Input array not resident on GPU. Call write() or ensure() first.")

        # Param 4: resident float output (already on GPU)
        buf_output = self.dmgr.buffers.get(id(output_array))
        if buf_output is None:
            raise ValueError(f"Output array not resident on GPU. Call write() or ensure() first.")

        # Execute kernel
        global_size = (input_array.size,) if self.autoflatten else input_array.shape
        kernel = self._kernels[f"{kernel_name}_res"]
        kernel(
            self.queue,
            global_size,
            None,  # local_size
            vertical_i32,
            width,
            height,
            buf_input,
            buf_output
        )
        # Note: No read() - buffers stay resident

    def gauss_h(
        self,
        kernel_name: str,
        input_array: np.ndarray,
        output_array: np.ndarray
    ) -> None:
        """
        Execute horizontal convolution pass with resident buffers.

        Interface:
            kernel ${name}_h(width_of input, height_of input,
                            resident float input, resident float output);

        Args:
            kernel_name: Name of kernel (e.g., 'gauss3', 'gauss5')
            input_array: Input image already on GPU
            output_array: Output buffer already on GPU
        """
        # Param 0: width_of input (derived)
        width = np.int32(input_array.shape[0])

        # Param 1: height_of input (derived)
        height = np.int32(input_array.shape[1])

        # Param 2: resident float input
        buf_input = self.dmgr.buffers.get(id(input_array))
        if buf_input is None:
            raise ValueError(f"Input array not resident on GPU")

        # Param 3: resident float output
        buf_output = self.dmgr.buffers.get(id(output_array))
        if buf_output is None:
            raise ValueError(f"Output array not resident on GPU")

        # Execute kernel
        global_size = (input_array.size,) if self.autoflatten else input_array.shape
        kernel = self._kernels[f"{kernel_name}_h"]
        kernel(
            self.queue,
            global_size,
            None,  # local_size
            width,
            height,
            buf_input,
            buf_output
        )

    def gauss_v(
        self,
        kernel_name: str,
        input_array: np.ndarray,
        output_array: np.ndarray
    ) -> None:
        """
        Execute vertical convolution pass with resident buffers.

        Interface:
            kernel ${name}_v(width_of input, height_of input,
                            resident float input, resident float output);

        Args:
            kernel_name: Name of kernel (e.g., 'gauss3', 'gauss5')
            input_array: Input image already on GPU
            output_array: Output buffer already on GPU
        """
        # Param 0: width_of input (derived)
        width = np.int32(input_array.shape[0])

        # Param 1: height_of input (derived)
        height = np.int32(input_array.shape[1])

        # Param 2: resident float input
        buf_input = self.dmgr.buffers.get(id(input_array))
        if buf_input is None:
            raise ValueError(f"Input array not resident on GPU")

        # Param 3: resident float output
        buf_output = self.dmgr.buffers.get(id(output_array))
        if buf_output is None:
            raise ValueError(f"Output array not resident on GPU")

        # Execute kernel
        global_size = (input_array.size,) if self.autoflatten else input_array.shape
        kernel = self._kernels[f"{kernel_name}_v"]
        kernel(
            self.queue,
            global_size,
            None,  # local_size
            width,
            height,
            buf_input,
            buf_output
        )


class GaussianKernelWrapper:
    """
    Wrapper that mimics the old Kernel interface but uses compiled package.

    This provides backward compatibility for code like:
        g = gaussian_kernel
        g.h(input, output)
        g.v(output, input)
    """

    def __init__(self, package: ConvolvesepPackage, kernel_name: str, width: int):
        self.package = package
        self.kernel_name = kernel_name
        self.width = width
        self.dmgr = package.dmgr

    def __call__(self, input_array: np.ndarray, vertical: int = 0) -> np.ndarray:
        """Main kernel call - copies data."""
        return self.package.gauss(self.kernel_name, vertical, input_array)

    def res(self, vertical: int, input_array: np.ndarray, output_array: np.ndarray) -> None:
        """Extension: resident buffer version."""
        self.package.gauss_res(self.kernel_name, vertical, input_array, output_array)

    def h(self, input_array: np.ndarray, output_array: np.ndarray) -> None:
        """Horizontal pass."""
        self.package.gauss_h(self.kernel_name, input_array, output_array)

    def v(self, input_array: np.ndarray, output_array: np.ndarray) -> None:
        """Vertical pass."""
        self.package.gauss_v(self.kernel_name, input_array, output_array)

    def write(self, array: np.ndarray, wait: bool = True):
        """Transfer array to GPU."""
        return self.dmgr.write(array, wait=wait)

    def read(self, array: np.ndarray):
        """Transfer array from GPU."""
        return self.dmgr.read(array)

    def ensure(self, array: np.ndarray):
        """Ensure array buffer exists on GPU."""
        return self.dmgr.ensure(array)
```

## Modified Gaussian Class (Gradual Migration)

```python
# yapocis/lib/gaussian.py

from yapocis.lib.bound_kernels import BoundKernels
from yapocis.utils.stereo_typing import *

# NEW: Import compiled package (optional)
try:
    from yapocis.kernelpack import get_convolvesep_package
    USE_COMPILED_PACKAGE = True
except ImportError:
    USE_COMPILED_PACKAGE = False


class Gaussian(BoundKernels):
    INTERFACE = "convolvesep"
    load_program_config = {}

    def __init__(
        self,
        device_name: str | None = None,
        gaussian_weights: list[np.ndarray] | None = None,
        use_package: bool = USE_COMPILED_PACKAGE  # NEW: Opt-in flag
    ):
        self.gaussian_weights = (
            GaussianWeights() if gaussian_weights is None else gaussian_weights
        )
        gws = self.gaussian_weights.gaussian_weights()
        convs = [("gauss%s" % len(a), a) for a in gws]

        super(Gaussian, self).__init__(device_name=device_name, convs=convs)
        kmgr = self.kmgr

        # NEW: Use compiled package if requested
        if use_package and USE_COMPILED_PACKAGE:
            self._init_with_package(convs)
        else:
            self._init_with_bound_kernels(convs)

        # Rest of initialization unchanged
        self.zc = ZC(device_name=device_name)
        self.grad = Gradient(device_name=device_name)
        self.ops = Operators(device_name=device_name)
        self.threshold = 7.5e-06

    def _init_with_package(self, convs):
        """Initialize using compiled kernel package (fast path)."""
        from yapocis.kernelpack import get_convolvesep_package

        # Get compiled package for current backend
        package = get_convolvesep_package(self.kmgr.dmgr)

        # Load all kernel variants
        kernel_names = [name for name, _ in convs]
        package.load(kernel_names)

        # Create wrapper objects that mimic old Kernel interface
        from yapocis.kernelpack.ocl.convolvesep import GaussianKernelWrapper

        self.gaussian_kernels = [
            GaussianKernelWrapper(package, name, len(conv))
            for name, conv in convs
        ]

        for i, (name, conv) in enumerate(convs):
            self.gaussian_kernels[i].width = len(conv)

    def _init_with_bound_kernels(self, convs):
        """Initialize using traditional runtime interpretation (slow path)."""
        kmgr = self.kmgr
        convsres = [("gauss%s_res" % len(a), a) for a in convs]
        convsh = [("gauss%s_h" % len(a), a) for a in convs]
        convsv = [("gauss%s_v" % len(a), a) for a in convs]

        self.gaussian_kernels = [
            getattr(kmgr, name) for (name, conv) in convs
        ]
        for i, (name, conv) in enumerate(convs):
            self.gaussian_kernels[i].res = getattr(kmgr, convsres[i][0])
            self.gaussian_kernels[i].h = getattr(kmgr, convsh[i][0])
            self.gaussian_kernels[i].v = getattr(kmgr, convsv[i][0])
            self.gaussian_kernels[i].width = len(conv)

    # Rest of class unchanged - works with either initialization!
    def gauss_image(self, a: Image1, scale: int) -> Image1:
        g = self.get_gaussian_kernel(scale)
        aa = a.copy()
        b = np.empty_like(aa)
        g.write(aa)
        g.write(b)
        g.h(aa, b)
        g.v(b, aa)
        g.read(aa)
        return aa
```

## Helper Module: `yapocis/kernelpack/__init__.py`

```python
"""
Kernel package loader - automatically selects backend.
"""

import os
from yapocis.core.manager import get_backend


def get_convolvesep_package(dmgr):
    """
    Get the compiled ConvolvesepPackage for current backend.

    Args:
        dmgr: Device manager (determines backend)

    Returns:
        ConvolvesepPackage instance for the appropriate backend
    """
    backend = get_backend()

    if backend == "ocl":
        from yapocis.kernelpack.ocl.convolvesep import ConvolvesepPackage
    elif backend == "metal":
        from yapocis.kernelpack.metal.convolvesep import ConvolvesepPackage
    elif backend == "cuda":
        from yapocis.kernelpack.cuda.convolvesep import ConvolvesepPackage
    elif backend == "numba":
        from yapocis.kernelpack.numba.convolvesep import ConvolvesepPackage
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return ConvolvesepPackage(dmgr, dmgr.program)
```

## Migration Path

### Phase 1: Generate Packages
```bash
# Generate for all backends
yapocis-package convolvesep --backend ocl --output yapocis/kernelpack/ocl/convolvesep.py
yapocis-package convolvesep --backend metal --output yapocis/kernelpack/metal/convolvesep.py
yapocis-package convolvesep --backend cuda --output yapocis/kernelpack/cuda/convolvesep.py
yapocis-package convolvesep --backend numba --output yapocis/kernelpack/numba/convolvesep.py
```

### Phase 2: Test with Gaussian (Opt-In)
```python
# Existing code works unchanged (slow path)
g = Gaussian()

# Opt in to compiled package (fast path)
g = Gaussian(use_package=True)

# Both work identically!
result = g.gauss_image(image, scale=2)
```

### Phase 3: Benchmark
```python
# Compare performance
import time

g_slow = Gaussian(use_package=False)
g_fast = Gaussian(use_package=True)

# Time 1000 iterations
image = np.random.rand(512, 512).astype(np.float32)

start = time.time()
for _ in range(1000):
    _ = g_slow.gauss_image(image, scale=2)
slow_time = time.time() - start

start = time.time()
for _ in range(1000):
    _ = g_fast.gauss_image(image, scale=2)
fast_time = time.time() - start

print(f"Speedup: {slow_time / fast_time:.2f}x")
```

### Phase 4: Make Default
Once validated, flip the default:
```python
def __init__(self, ..., use_package: bool = True):  # Changed default
```

### Phase 5: Deprecate Old Path
After several releases, remove runtime interpretation entirely.

## Implementation Tasks

1. **Create directory structure**
   - [ ] `yapocis/kernelpack/__init__.py`
   - [ ] `yapocis/kernelpack/ocl/__init__.py`
   - [ ] `yapocis/kernelpack/metal/__init__.py`
   - [ ] `yapocis/kernelpack/cuda/__init__.py`
   - [ ] `yapocis/kernelpack/numba/__init__.py`

2. **Create code generator**
   - [ ] `yapocis/codegen/package_generator.py`
   - [ ] Mako templates for each backend
   - [ ] CLI tool: `yapocis-package`

3. **Generate convolvesep packages**
   - [ ] OCL backend
   - [ ] Metal backend
   - [ ] CUDA backend
   - [ ] Numba backend

4. **Modify Gaussian class**
   - [ ] Add `use_package` parameter
   - [ ] Implement `_init_with_package()`
   - [ ] Keep `_init_with_bound_kernels()` for compatibility

5. **Testing**
   - [ ] All existing Gaussian tests pass with `use_package=False`
   - [ ] All existing Gaussian tests pass with `use_package=True`
   - [ ] Benchmark performance improvement
   - [ ] Test all backends (OCL, Metal, CUDA, Numba)

6. **Documentation**
   - [ ] Update CLAUDE.md with package system
   - [ ] Add migration guide
   - [ ] Performance comparison docs

## Success Criteria

- ✅ Zero existing tests break
- ✅ `Gaussian(use_package=False)` works identically to current behavior
- ✅ `Gaussian(use_package=True)` produces same numerical results
- ✅ 2-5x performance improvement in kernel call overhead
- ✅ All 4 backends supported
- ✅ Code passes `make lint` and `make typecheck`
