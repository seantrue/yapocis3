# Kernel Packager Design Document

## Goal

Generate Python classes from interface specifications that eliminate runtime interpretation overhead by pre-compiling all buffer management logic into direct method calls.

## Current State (Runtime Interpretation)

Currently, when you call a kernel:

```python
# Runtime interpretation happens in Kernel.__call__()
result = prog.mandelbrot(grid, maxiter=256)
```

The call goes through:
1. `Kernel.prepare_arguments()` - interprets parameter hints at runtime
2. Dynamic dispatch to `handle_in()`, `handle_out_like()`, etc.
3. Type checking and coercion
4. Buffer allocation/transfer
5. Kernel invocation
6. Return value collection

This happens **every call** with string lookups, type checks, and dynamic dispatch.

## Proposed Solution: Compiled Kernel Packages

Generate a Python class with **typed, pre-compiled methods** for each kernel:

```python
class MandelbrotPackage:
    """Auto-generated from mandelbrot.interface"""

    def __init__(self, dmgr: DeviceManager, program: pyopencl.Program):
        self.dmgr = dmgr
        self.queue = dmgr.queue
        self._mandelbrot = program.mandelbrot

    def mandelbrot(self, q: np.ndarray, maxiter: int) -> np.ndarray:
        """
        kernel mandelbrot(in complex64 q, out_like short q, int maxiter);

        Args:
            q: complex64 input array
            maxiter: int32 scalar

        Returns:
            short array with same shape as q
        """
        # Pre-compiled buffer management (no interpretation!)

        # Param 0: in complex64 q
        q = q.astype(np.complex64, copy=False)
        buf_q = self.dmgr.write(q, wait=True)

        # Param 1: out_like short q (return value)
        out = np.empty_like(q, dtype=np.int16)
        buf_out = self.dmgr.ensure(out)

        # Param 2: int maxiter (scalar)
        maxiter_i32 = np.int32(maxiter)

        # Determine global size from first array argument
        global_size = (q.size,) if self.dmgr.autoflatten else q.shape

        # Execute kernel
        self._mandelbrot(
            self.queue,
            global_size,
            None,  # local_size
            buf_q,
            buf_out,
            maxiter_i32
        )

        # Collect return values
        self.dmgr.read(out)
        return out
```

## Benefits

1. **Performance**: No runtime interpretation, type checking, or string lookups
2. **Type Safety**: Full type hints for IDE autocomplete and mypy checking
3. **Transparency**: Generated code is human-readable Python
4. **Debugging**: Stack traces show actual code, not framework internals
5. **Optimization**: Can specialize per-backend (OCL vs CUDA vs Metal)

## Code Generator Architecture

```
yapocis/codegen/
├── __init__.py
├── package_generator.py    # Main generator class
├── templates/
│   ├── package_class.py.mako      # Class template
│   ├── kernel_method.py.mako      # Method template
│   └── buffer_handling.py.mako    # Buffer ops snippets
└── cli.py                   # CLI tool
```

### Generator Class Structure

```python
class KernelPackageGenerator:
    """Generate Python package classes from interface specs."""

    def __init__(self, backend: str = "ocl"):
        self.backend = backend
        self.templates = self._load_templates()

    def generate_package(
        self,
        interface_name: str,
        output_path: Path | None = None
    ) -> str:
        """
        Generate a KernelPackage class from an interface.

        Args:
            interface_name: Name of interface (e.g., "mandelbrot")
            output_path: If provided, write to file; else return string

        Returns:
            Generated Python source code
        """
        interface = self._load_interface(interface_name)
        code = self._generate_class(interface)

        if output_path:
            output_path.write_text(code)

        return code

    def _generate_class(self, interface: InterfaceSpec) -> str:
        """Generate complete class from interface spec."""
        methods = []
        for kernel in interface.kernels:
            if not kernel.alias:  # Skip aliases, generate separately
                methods.append(self._generate_kernel_method(kernel))

        # Generate extension methods (like _res variants)
        for kernel in interface.kernels:
            if kernel.alias:
                methods.append(self._generate_extension_method(kernel))

        return self.templates['class'].render(
            interface=interface,
            methods=methods,
            backend=self.backend
        )

    def _generate_kernel_method(self, kernel: KernelSpec) -> str:
        """Generate a single kernel method with compiled buffer logic."""

        # Analyze parameters to determine operations
        ops = self._analyze_parameters(kernel.parameters)

        return self.templates['method'].render(
            kernel=kernel,
            operations=ops,
            backend=self.backend
        )

    def _analyze_parameters(
        self,
        params: list[KernelParamSpec]
    ) -> list[dict]:
        """
        Analyze parameters and return compiled operations.

        Returns list of operations like:
        [
            {
                'type': 'buffer_in',
                'name': 'q',
                'dtype': 'complex64',
                'np_dtype': 'np.complex64',
                'derived_from': 'q'
            },
            {
                'type': 'buffer_out_like',
                'name': 'out',
                'dtype': 'int16',
                'np_dtype': 'np.int16',
                'derived_from': 'q',
                'shape_from': 'q'
            },
            {
                'type': 'scalar',
                'name': 'maxiter',
                'dtype': 'int32',
                'np_dtype': 'np.int32'
            }
        ]
        """
        operations = []

        for param in params:
            if param.hint == 'in':
                operations.append({
                    'type': 'buffer_in',
                    'param': param,
                    'code': self._gen_buffer_in(param)
                })
            elif param.hint == 'out_like':
                operations.append({
                    'type': 'buffer_out_like',
                    'param': param,
                    'code': self._gen_buffer_out_like(param)
                })
            elif param.hint == 'out':
                operations.append({
                    'type': 'buffer_out',
                    'param': param,
                    'code': self._gen_buffer_out(param)
                })
            elif param.hint == 'inout':
                operations.append({
                    'type': 'buffer_inout',
                    'param': param,
                    'code': self._gen_buffer_inout(param)
                })
            elif param.hint == 'resident':
                operations.append({
                    'type': 'buffer_resident',
                    'param': param,
                    'code': self._gen_buffer_resident(param)
                })
            elif param.dim:  # width_of, height_of, size_of, depth_of
                operations.append({
                    'type': 'derived_dimension',
                    'param': param,
                    'code': self._gen_dimension(param)
                })
            else:  # scalar
                operations.append({
                    'type': 'scalar',
                    'param': param,
                    'code': self._gen_scalar(param)
                })

        return operations
```

## Generated Code Examples

### Example 1: Simple Kernel (mandelbrot)

**Interface:**
```idl
interface mandelbrot {
    kernel mandelbrot(in complex64 q, out_like short q, int maxiter);
};
```

**Generated:**
```python
class MandelbrotPackage:
    def __init__(self, dmgr, program):
        self.dmgr = dmgr
        self.queue = dmgr.queue
        self._mandelbrot = program.mandelbrot

    def mandelbrot(self, q: np.ndarray, maxiter: int) -> np.ndarray:
        # in complex64 q
        q = q.astype(np.complex64, copy=False)
        buf_q = self.dmgr.write(q, wait=True)

        # out_like short q
        out = np.empty_like(q, dtype=np.int16)
        buf_out = self.dmgr.ensure(out)

        # int maxiter
        maxiter = np.int32(maxiter)

        # Execute
        global_size = (q.size,) if self.dmgr.autoflatten else q.shape
        self._mandelbrot(self.queue, global_size, None, buf_q, buf_out, maxiter)

        # Return
        self.dmgr.read(out)
        return out
```

### Example 2: Kernel with Extensions (graphics4.lines)

**Interface:**
```idl
kernel lines(width_of a, height_of a, inout float a, width_of lines, in float lines, in float distance, in float rcolor);
extension res as lines(width_of a, height_of a, resident float a, width_of lines, in float lines, in float distance, in float rcolor);
```

**Generated:**
```python
class Graphics4Package:
    def __init__(self, dmgr, program):
        self.dmgr = dmgr
        self.queue = dmgr.queue
        self._lines = program.lines

    def lines(
        self,
        a: np.ndarray,
        lines: np.ndarray,
        distance: float,
        rcolor: np.ndarray
    ) -> np.ndarray:
        # width_of a (derived)
        width = np.int32(a.shape[0])

        # height_of a (derived)
        height = np.int32(a.shape[1])

        # inout float a
        a = a.astype(np.float32, copy=False)
        buf_a = self.dmgr.write(a, wait=True)

        # width_of lines (derived)
        width_lines = np.int32(lines.shape[0])

        # in float lines
        lines = lines.astype(np.float32, copy=False)
        buf_lines = self.dmgr.write(lines, wait=True)

        # in float distance
        distance = np.float32(distance)

        # in float rcolor
        rcolor = rcolor.astype(np.float32, copy=False)
        buf_rcolor = self.dmgr.write(rcolor, wait=True)

        # Execute
        global_size = (a.size,) if self.dmgr.autoflatten else a.shape
        self._lines(
            self.queue, global_size, None,
            width, height, buf_a, width_lines, buf_lines, distance, buf_rcolor
        )

        # Return inout buffers
        self.dmgr.read(a)
        return a

    def lines_res(
        self,
        a: np.ndarray,  # Already resident on GPU
        lines: np.ndarray,
        distance: float,
        rcolor: np.ndarray
    ) -> None:  # No return - modifies resident buffer
        # width_of a
        width = np.int32(a.shape[0])

        # height_of a
        height = np.int32(a.shape[1])

        # resident float a (already on GPU - use buffer directly)
        buf_a = self.dmgr.buffers.get(id(a))
        if buf_a is None:
            raise ValueError(f"Array 'a' is not resident on GPU")

        # width_of lines
        width_lines = np.int32(lines.shape[0])

        # in float lines
        lines = lines.astype(np.float32, copy=False)
        buf_lines = self.dmgr.write(lines, wait=True)

        # in float distance
        distance = np.float32(distance)

        # in float rcolor
        rcolor = rcolor.astype(np.float32, copy=False)
        buf_rcolor = self.dmgr.write(rcolor, wait=True)

        # Execute
        global_size = (a.size,) if self.dmgr.autoflatten else a.shape
        self._lines(
            self.queue, global_size, None,
            width, height, buf_a, width_lines, buf_lines, distance, buf_rcolor
        )
        # Note: No read() call - buffer stays resident
```

## CLI Tool

```bash
# Generate package for a single interface
yapocis-package mandelbrot --output yapocis/packages/mandelbrot.py

# Generate packages for all interfaces
yapocis-package --all --output-dir yapocis/packages/

# Generate for specific backend
yapocis-package graphics4 --backend metal --output graphics4_metal.py

# Preview without writing
yapocis-package mandelbrot --dry-run
```

## Integration with Existing Code

### Option 1: Replace BoundKernels

```python
# Current way (runtime interpretation)
from yapocis.lib.bound_kernels import BoundKernels

class Gaussian(BoundKernels):
    INTERFACE = "convolvesep"
    # ...

# New way (compiled)
from yapocis.packages.convolvesep import ConvolvesepPackage

class Gaussian:
    def __init__(self, device_name=None):
        mgr = get_manager()
        prog = core.load_program("convolvesep", ...)
        self.pkg = ConvolvesepPackage(mgr.dmgr, prog)

    def gauss_image(self, a, scale):
        # Direct calls - no interpretation!
        self.pkg.gauss19_h(a, ...)
        self.pkg.gauss19_v(a, ...)
```

### Option 2: Parallel Use (Gradual Migration)

Keep both approaches available:

```python
# Fast path - use compiled package
from yapocis.packages import MandelbrotPackage

pkg = MandelbrotPackage(dmgr, program)
result = pkg.mandelbrot(grid, maxiter=256)

# Slow path - use runtime interpretation (backward compat)
import core
prog = core.load_program("mandelbrot")
result = prog.mandelbrot(grid, maxiter=256)
```

## Implementation Plan

### Phase 1: Core Generator (Week 1)
- [ ] Create `yapocis/codegen/` package structure
- [ ] Implement `KernelPackageGenerator` class
- [ ] Create Mako templates for:
  - Class structure
  - Method signatures
  - Buffer operations (in, out, inout, out_like, resident)
  - Scalar handling
  - Derived dimensions (width_of, height_of, etc.)
- [ ] Add CLI tool `yapocis-package`
- [ ] Unit tests for generator

### Phase 2: Template Completeness (Week 1-2)
- [ ] Handle all parameter hints:
  - `in`, `out`, `inout`, `out_like`
  - `resident` (GPU-resident buffers)
  - Derived: `width_of`, `height_of`, `size_of`, `depth_of`
- [ ] Handle extensions/aliases correctly
- [ ] Generate proper type hints
- [ ] Add docstrings to generated methods

### Phase 3: Backend Support (Week 2)
- [ ] OpenCL backend code generation
- [ ] CUDA backend code generation (different buffer API)
- [ ] Metal backend code generation
- [ ] Numba backend code generation

### Phase 4: Integration & Testing (Week 3)
- [ ] Generate packages for all existing interfaces
- [ ] Performance benchmarks (compiled vs interpreted)
- [ ] Migrate one high-level lib (e.g., Gaussian) to use packages
- [ ] Documentation and examples

### Phase 5: Advanced Features (Week 4)
- [ ] Support for tiled kernels
- [ ] Local memory handling
- [ ] Async operations
- [ ] Batch operations
- [ ] Auto-tuning integration

## Open Questions

1. **Package location**: Generate to `yapocis/packages/` or user code?
2. **Regeneration**: When to regenerate? Build step? On-demand?
3. **Version tracking**: How to track interface version vs generated code?
4. **Error messages**: How to provide good errors in generated code?
5. **Debugging**: Source maps? Line number mapping?

## Success Metrics

1. **Performance**: 2-5x speedup for kernel invocation overhead
2. **Type Safety**: 100% of methods have complete type hints
3. **Coverage**: All 12 existing interfaces generate valid packages
4. **Compatibility**: All existing tests pass with generated packages
5. **Code Quality**: Generated code passes `make lint` and `make typecheck`
