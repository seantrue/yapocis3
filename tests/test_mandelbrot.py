import numpy as np
import pytest

from . import StopWatcher


@pytest.mark.track_memory
def test_mandelbrot(perf_tracker):
    """Test Mandelbrot set computation with performance tracking."""
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
