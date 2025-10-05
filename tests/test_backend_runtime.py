"""
Runtime sanity tests for each backend executing a simple demo kernel.

Each backend runs the 'demo.sum' kernel on a tiny vector and validates output.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.ocl
def test_runtime_demo_ocl(monkeypatch):
    try:
        import pyopencl  # noqa: F401
    except Exception as e:
        pytest.skip(f"pyopencl not available: {e}")
    monkeypatch.setenv("YAPOCIS_BACKEND", "ocl")
    from yapocis.core.manager import get_manager

    mgr = get_manager("ocl")
    prog = mgr.load_program("demo")
    a = np.arange(16, dtype=np.float32)
    b = np.ones_like(a) * 5
    out = prog.sum(a, b, global_size=a.size)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, a + b)


@pytest.mark.metal
def test_runtime_demo_metal(monkeypatch):
    try:
        import pymetallic  # noqa: F401
    except Exception as e:
        pytest.skip(f"pymetallic not available: {e}")
    monkeypatch.setenv("YAPOCIS_BACKEND", "metal")
    from yapocis.core.manager import get_manager

    mgr = get_manager("metal")
    prog = mgr.load_program("demo")
    a = np.arange(16, dtype=np.float32)
    b = np.ones_like(a) * 7
    out = prog.sum(a, b, global_size=a.size)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, a + b)


@pytest.mark.cuda
def test_runtime_demo_cuda(monkeypatch):
    try:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda  # noqa: F401
    except Exception as e:
        pytest.skip(f"pycuda not available: {e}")
    monkeypatch.setenv("YAPOCIS_BACKEND", "cuda")
    from yapocis.core.manager import get_manager

    mgr = get_manager("cuda")
    prog = mgr.load_program("demo")

    # Use numpy arrays; CUDA wrapper will stage to device and return host array
    a = np.arange(16, dtype=np.float32)
    b = np.ones_like(a) * 11
    out = prog.sum(a, b, global_size=int(a.size))
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, a + b)


@pytest.mark.numba
def test_runtime_demo_numba(monkeypatch):
    try:
        import numba  # noqa: F401
    except Exception as e:
        pytest.skip(f"numba not available: {e}")
    monkeypatch.setenv("YAPOCIS_BACKEND", "numba")
    from yapocis.core.manager import get_manager

    mgr = get_manager("numba")
    prog = mgr.load_program("demo")
    a = np.arange(16, dtype=np.float32)
    b = np.ones_like(a) * 3
    out = prog.sum(a, b, global_size=a.size)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, a + b)
