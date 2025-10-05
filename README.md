# What is Yapocis?

Yapocis is "Yet another Python OpenCL interface specification" — now with multiple backends: OpenCL, CUDA, Metal, and Numba.

Memorable. Easy to pronounce and spell. Clearly related to the semantics of the project.

Or not, all the good names are used and I am not in marketing.

It includes easy-to-call-from-Python, reasonably performing implementations of kernels such as Mandelbrot, 3x3 median, image gradients, HSI <-> RGB conversions, color math, 2D graphics primitives, and signed distance field path tracing. I use it to develop much less generally useful implementations of bizarre image processing ideas.

## Why this happened?

*Some brief history, "living backwards" order:*

I'm putting this code up on Github to be used or snickered at. Both can be helpful. It has been used on macOS and Linux with a 64-bit Python, and assumes NumPy and Matplotlib for examples/demos. It supports OpenCL (PyOpenCL), CUDA (CuPy), Metal (via a thin `pymetallic` shim), and Numba (CPU fallback with JIT compilation). I can't help resolve issues on dependencies; it's hard enough to make new ones happen.

*This in response to a note I got back from the pyopencl list:*

Sounds interesting. Please let the list know once you have public code to play with.

Andreas

*Which was in response to the following inquiry I sent to the list:*

I'm wondering if yet-another-python-opencl-interface-layer would be of interest. This layer is built on top of pyopencl, and is intended to use familiar RPC definitions to ease the special joys of talking to opencl.

I've implemented this on OS/X and it is working for me and has been stable, oh, for several days (insert smiley face here). It eliminates the need for routine glue code, and can be extended to minimize movement of data between host and opencl service.

This code is currently private, and I will be happy to either fork pyopencl and submit a pull after integration, or a separate github project.

-- Sean

> Sean True  
> Swapwizard Consulting  
> *Not my day job*

I've been trying to make these algorithms run fast enough to be useful and easy to use enough to make more core ideas testable for, oh, 25 years. Now I can work on camera resolution images in relatively quick turn around.

## Presumed highlights

```python
# Use the rpc extension to define and load the kernel as a callable.
from yapocis.rpc import kernels, interfaces
calc_fractal_opencl = kernels.loadProgram(interfaces.mandelbrot).mandelbrot
# Call it the way we like to call Python callables:
output = calc_fractal(q, maxiter)
```

```text
# RPC definition language loosely based on Apollo NCS/OSF DCE/Microsoft IDL
# outlike is a novel keyword that says: allocate for me, return as part
# of return vals.

interface mandelbrot {
      kernel mandelbrot(in complex64 *q, outlike int16 *q, in int32 maxiter);
      };
```

```c
// mandelbrot.mako is just what it has always been (OpenCL). Metal/CUDA versions are auto-translated templates.
__kernel void mandelbrot(__global float2 *q,  __global short *output, int const maxiter)
{
    int gid = get_global_id(0);
    float nreal, real = 0;
    float imag = 0;
    output[gid] = 0;
    for(int curiter = 0; curiter < maxiter; curiter++) {
            nreal = real*real - imag*imag + q[gid].x;
            imag = 2* real*imag + q[gid].y;
            real = nreal;
            if (real*real + imag*imag > 4.0f)
              output[gid] = curiter;
    }
}
```

## Backends and selection

- Backends: `ocl` (OpenCL via PyOpenCL), `cuda` (CuPy), `metal` (PyMetallic shim), and `numba` (CPU with JIT).
- Select backend via env: `export YAPOCIS_BACKEND=ocl|cuda|metal|numba`.
- You can also select explicitly in code:

```python
from yapocis.core.manager import get_manager
mgr = get_manager("metal")  # or "ocl"/"cuda"/"numba"
prog = mgr.load_program("demo")
```

High-level wrappers (e.g., `yapocis.lib.*`) route through `yapocis.core.manager` and honor `YAPOCIS_BACKEND`.

**Numba backend notes:**
- Provides CPU fallback when GPU backends are unavailable
- Uses Numba's JIT compilation for Python/NumPy code
- Automatically selected when no GPU backend is available
- Useful for testing and development without GPU hardware

## Development & tests

Quickstart:

```bash
# Using make (recommended)
make install-dev

# Or manually with uv
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Install with all backends
make install-backends

# Run tests
make test

# Run backend-specific tests
make test-ocl    # OpenCL tests
make test-metal  # Metal tests
make test-numba  # Numba tests
```

Backend test markers and dependencies:

- Markers exist for each backend: `-m ocl`, `-m metal`, `-m cuda`, `-m numba`.
- Tests will skip cleanly if the backend runtime isn't installed:
  - OpenCL tests require `pyopencl`.
  - Metal tests require a working `pymetallic` runtime and an Apple GPU.
  - CUDA tests require `cupy` for the core CUDA runtime tests; the PyCUDA legacy test requires `pycuda`.
  - Numba tests require `numba` (usually available in most environments).
- Useful invocations:
  - `pytest -q -m metal` to run only Metal tests.
  - `pytest -q -m ocl` to run only OpenCL tests.
  - `pytest -q -m cuda` to run only CUDA tests.
  - `pytest -q -m numba` to run only Numba tests.
  - `pytest -q tests/test_backend_smoke.py` to smoke‑compile all interfaces across available backends.
  - `pytest -q tests/test_backend_runtime.py` to run a tiny runtime sanity call per backend.

CUDA/PyCUDA notes:

- The CUDA demo/runtime tests use CuPy and do not require PyCUDA.
- The legacy PyCUDA test at `yapocis/cukernels/test_convolve.py` is auto‑skipped if `pycuda` isn’t installed, so you can still run the rest of the suite.

## Known misfeatures

There are several. Here are some I remember:

- Untested in any other environment, as noted.
- CUDA tests under `yapocis/cukernels` depend on `pycuda`. They are excluded from standard test runs.

## Changes and new features

- Reducing data movement between engine and Python code. This is implemented using an alias keyword to add alternate signatures to a kernel, and a resident parameter hint which hints the glue code to just use the buffer as-is on the engine. V0.2.2
- Framework for dealing with two-dimensional arrays auto-magically. New buffer hints: sizeof, widthof, heightof for setting arguments invisibly.
- The OpenCL engine to use is configured in `rpc.kernels`. Developer has choice of `CPU_ENGINE` and `GPU_ENGINE`.

## Future features

Other than removing the misfeatures:

- Backend-aware scheduling strategies (hardware and array-geometry sensitive execution).
