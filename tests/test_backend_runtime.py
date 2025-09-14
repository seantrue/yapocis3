"""
Runtime sanity tests for each backend executing a simple demo kernel.

Each backend runs the 'demo.sum' kernel on a tiny vector and validates output.
"""

from __future__ import annotations

import os
import numpy as np
import pytest


def _set_backend(backend: str):
    os.environ["YAPOCIS_BACKEND"] = backend


@pytest.mark.ocl
def test_runtime_demo_ocl():
    try:
        import pyopencl  # noqa: F401
    except Exception as e:
        pytest.skip(f"pyopencl not available: {e}")
    _set_backend("ocl")
    from yapocis.core.manager import get_manager
    mgr = get_manager("ocl")
    prog = mgr.load_program("demo")
    a = np.arange(16, dtype=np.float32)
    b = np.ones_like(a) * 5
    out = prog.sum(a, b, global_size=a.size)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, a + b)


@pytest.mark.metal
def test_runtime_demo_metal():
    try:
        import pymetallic  # noqa: F401
    except Exception as e:
        pytest.skip(f"pymetallic not available: {e}")
    _set_backend("metal")
    from yapocis.core.manager import get_manager
    mgr = get_manager("metal")
    prog = mgr.load_program("demo")
    a = np.arange(16, dtype=np.float32)
    b = np.ones_like(a) * 7
    out = prog.sum(a, b, global_size=a.size)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, a + b)


@pytest.mark.cuda
def test_runtime_demo_cuda():
    try:
        import cupy as cp  # type: ignore
    except Exception as e:
        pytest.skip(f"cupy not available: {e}")
    _set_backend("cuda")
    from yapocis.core.manager import get_manager
    mgr = get_manager("cuda")
    prog = mgr.load_program("demo")

    # Use cupy device arrays so we can observe in-place mutation
    a = cp.arange(16, dtype=cp.float32)
    b = cp.ones_like(a) * 11
    c = cp.zeros_like(a)
    # CUDA wrapper does not auto-allocate output; pass c explicitly
    _ = prog.sum(a, b, c, global_size=int(a.size))
    out = cp.asnumpy(c)
    np.testing.assert_allclose(out, cp.asnumpy(a + b))

