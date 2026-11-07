import numpy as np
import pytest

from yapocis.core import get_test_backends

from . import StopWatcher


@pytest.mark.track_memory
@pytest.mark.smoke
@pytest.mark.parametrize("backend", get_test_backends())
def test_mandelbrot(backend, perf_tracker, monkeypatch):
    """Test Mandelbrot set computation with performance tracking."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    from yapocis.lib.mandelbrot import Mandelbrot

    mb = Mandelbrot()
    # Build a small complex plane and a max iteration
    H, W = 512, 512
    re = np.linspace(-2.0, 1.0, W, dtype=np.float32)
    im = np.linspace(-1.5, 1.5, H, dtype=np.float32)
    xx, yy = np.meshgrid(re, im, indexing="xy")
    q = (xx + 1j * yy).astype(np.complex64)
    maxiter = 50

    with StopWatcher("mandelbrot", kernel=mb) as sw:
        with sw.checkpoint("mandelbrot_kernel"):
            # Time the main computation
            out = perf_tracker.time_operation(
                "mandelbrot_computation", lambda: mb.mandelbrot(q, maxiter)
            )

        with sw.checkpoint("normalization"):
            # Time the post-processing
            out = perf_tracker.time_operation(
                "normalize_result", lambda: out.astype(np.float32) / out.max()
            )

        sw.show(out)


@pytest.mark.parametrize("backend", [b for b in get_test_backends() if b != "numbax"])
def test_mandelbrot_records_the_first_escape_as_the_numba_port_does(backend, monkeypatch):
    """The device kernels stop at the first escape, as the numba port does. They differ only
    where float arithmetic decides the escape (|z|^2 > 4 against numpy's |z| > nextafter(2),
    with or without fused multiply-add): a few pixels in a thousand at most."""
    import importlib

    import yapocis.lib.mandelbrot as module

    re = np.linspace(-2.0, 1.0, 512, dtype=np.float32)
    im = np.linspace(-1.5, 1.5, 512, dtype=np.float32)
    xx, yy = np.meshgrid(re, im, indexing="xy")
    q = (xx + 1j * yy).astype(np.complex64)
    out = {}
    for b in ("numbax", backend):
        monkeypatch.setenv("YAPOCIS_BACKEND", b)
        importlib.reload(module)
        out[b] = np.asarray(module.Mandelbrot().mandelbrot(q, 50)).astype(np.int32)
    differ = out[backend] != out["numbax"]
    assert differ.mean() < 1e-3, f"{differ.mean():.5f} of pixels differ"
