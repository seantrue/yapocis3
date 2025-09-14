# Testing Backends

Yapocis ships helper tests to validate that interfaces compile and a small demo kernel runs across supported backends.

- Backends: `ocl` (OpenCL via PyOpenCL), `cuda` (PyCUDA), `metal` (PyMetallic shim)
- Select backend globally with `YAPOCIS_BACKEND` or per test via markers.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements_dev.txt
pip install -e .
```

## Smoke Compilation

Smoke-compile every interface for available backends (skips if runtime missing):

```bash
pytest -q tests/test_backend_smoke.py
```

What it does:
- Iterates `yapocis/interfaces/*.interface`
- Supplies minimal Mako context for parameterized interfaces (e.g., `operators`, `convolve*`)
- Tries to render and compile kernels for `ocl`, `cuda`, and `metal`

## Runtime Sanity

Run a tiny demo kernel (`demo.sum`) per backend and validate output:

```bash
pytest -q tests/test_backend_runtime.py
```

Notes:
- OpenCL and Metal tests use NumPy arrays and return host arrays.
- CUDA runtime tests use PyCUDA; numpy arrays are staged to device and results are returned to host.

## Backend Markers

Use markers to run backend-specific subsets:

```bash
pytest -q -m metal   # only Metal tests
pytest -q -m ocl     # only OpenCL tests
pytest -q -m cuda    # only CUDA tests
```

## Dependencies and Skips

- OpenCL tests require `pyopencl`
- Metal tests require `pymetallic` and an Apple GPU
- CUDA runtime tests require `pycuda`
- Legacy PyCUDA test under `yapocis/cukernels/test_convolve.py` is auto-skipped if `pycuda` is not installed

## Device Selection

- OpenCL device selection is handled by the OpenCL backend config
- Metal device can be set via `YAPOCIS_ENGINE` (name or index as reported by `pymetallic`)
- CUDA uses the default device via PyCUDA's context (pycuda.autoinit)
