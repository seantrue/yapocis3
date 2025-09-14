import os
import numpy as np
import pytest


@pytest.mark.parametrize("backend", ["metal"])  # sanity: backend route works
def test_template_resolution_and_runtime_sum(backend):
    try:
        import pymetallic  # noqa: F401
    except Exception as e:
        pytest.skip(f"pymetallic not available: {e}")

    from yapocis.core.manager import get_manager

    os.environ["YAPOCIS_BACKEND"] = backend
    mgr = get_manager("metal")
    prog = mgr.load_program("demo")
    # Source must come from metalkernels/demo.mako
    assert "kernel void sum" in prog.source

    # Call sum on simple arrays and verify results
    a = np.arange(16, dtype=np.float32)
    b = np.ones_like(a) * 3
    out = prog.sum(a, b, global_size=a.size)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, a + b)
