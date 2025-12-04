"""
Runtime sanity tests for each backend executing a simple demo kernel.

Each backend runs the 'demo.sum' kernel on a tiny vector and validates output.
"""

from __future__ import annotations

import numpy as np
import pytest

from yapocis.core import get_manager, load_program


@pytest.mark.smoke
@pytest.mark.ocl
def test_runtime_demo_ocl(monkeypatch):
    try:
        import pyopencl  # noqa: F401
    except Exception as e:
        pytest.skip(f"pyopencl not available: {e}")
    monkeypatch.setenv("YAPOCIS_BACKEND", "ocl")

    mgr = get_manager("ocl")
    prog = load_program("demo")
    a = np.arange(16, dtype=np.float32)
    b = np.ones_like(a) * 5
    out = prog.sum(a, b, global_size=a.size)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, a + b)


@pytest.mark.smoke
@pytest.mark.metal
def test_runtime_demo_metal(monkeypatch):
    try:
        import pymetallic  # noqa: F401
    except Exception as e:
        pytest.skip(f"pymetallic not available: {e}")
    monkeypatch.setenv("YAPOCIS_BACKEND", "metal")

    mgr = get_manager("metal")
    prog = load_program("demo")
    a = np.arange(16, dtype=np.float32)
    b = np.ones_like(a) * 7
    out = prog.sum(a, b, global_size=a.size)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, a + b)


@pytest.mark.smoke
@pytest.mark.cuda
def test_runtime_demo_cuda(monkeypatch):
    try:
        # import pycuda.autoinit  # noqa: F401
        # import pycuda.driver as cuda  # noqa: F401
        pass
    except Exception as e:
        pytest.skip(f"pycuda not available: {e}")
    monkeypatch.setenv("YAPOCIS_BACKEND", "cuda")

    # mgr = get_manager("cuda")
    prog = load_program("demo")

    # Use numpy arrays; CUDA wrapper will stage to device and return host array
    a = np.arange(16, dtype=np.float32)
    b = np.ones_like(a) * 11
    out = prog.sum(a, b, global_size=int(a.size))
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, a + b)


@pytest.mark.smoke
@pytest.mark.numba
def test_runtime_demo_numba(monkeypatch):
    try:
        import numba  # noqa: F401
    except Exception as e:
        pytest.skip(f"numba not available: {e}")
    monkeypatch.setenv("YAPOCIS_BACKEND", "numbax")
    from yapocis.core import load_program

    mgr = get_manager("numbax")
    prog = load_program("demo")
    a = np.arange(16, dtype=np.float32)
    b = np.ones_like(a) * 3
    out = prog.sum(a, b, global_size=a.size)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, a + b)
