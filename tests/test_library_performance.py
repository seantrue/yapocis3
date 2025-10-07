#!/usr/bin/env python
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.
"""
Comprehensive performance testing for all yapocis library entry points.

Tests every kernel and library function across all available backends and devices,
saving performance data to the DuckDB database for analysis.
"""
import os

import numpy as np
import pytest

from yapocis.core.manager import get_backends, get_manager

# Import library classes that we'll test
from yapocis.lib.anisodiff import Anisodiff
from yapocis.lib.gaussian import Gaussian
from yapocis.lib.gradient import Gradient
from yapocis.lib.graphics4 import Graphics4
from yapocis.lib.hsi import HSI
from yapocis.lib.mandelbrot import Mandelbrot
from yapocis.lib.median import Median
from yapocis.lib.operators import Operators
from yapocis.lib.truetrace import TrueTrace
from yapocis.lib.zc import ZC

from tests import StopWatcher


# Test data sizes - keep small for fast performance testing
SMALL_SIZE = 256
MEDIUM_SIZE = 512
REPEATS = 3  # Number of times to repeat each operation for consistent timing


@pytest.mark.parametrize("backend", get_backends())
class TestKernelPerformance:
    """Test raw kernel performance across all interfaces."""

    def test_operators_performance(self, backend):
        """Test all operator kernels: add, sub, mul, divop."""
        os.environ["YAPOCIS_BACKEND"] = backend

        ops = Operators()

        # Test data
        a = np.random.uniform(0, 10, SMALL_SIZE * SMALL_SIZE).astype(np.float32)
        b = np.random.uniform(1, 10, SMALL_SIZE * SMALL_SIZE).astype(np.float32)

        with StopWatcher("operators", save_perf=True, kernel=ops) as sw:
            for _ in range(REPEATS):
                with sw.checkpoint("add"):
                    _ = ops.add(a, b)

                with sw.checkpoint("sub"):
                    _ = ops.sub(a, b)

                with sw.checkpoint("mul"):
                    _ = ops.mul(a, b)

                with sw.checkpoint("divop"):
                    _ = ops.divop(a, b)

        # Verify one operation worked
        result = ops.add(a, b)
        assert result.shape == a.shape

    def test_mandelbrot_performance(self, backend):
        """Test mandelbrot kernel."""
        os.environ["YAPOCIS_BACKEND"] = backend

        mb = Mandelbrot()

        # Complex grid
        H, W = SMALL_SIZE, SMALL_SIZE
        re = np.linspace(-2.0, 1.0, W, dtype=np.float32)
        im = np.linspace(-1.5, 1.5, H, dtype=np.float32)
        xx, yy = np.meshgrid(re, im, indexing="xy")
        q = (xx + 1j * yy).astype(np.complex64)
        maxiter = 50

        with StopWatcher("mandelbrot", save_perf=True, kernel=mb) as sw:
            for _ in range(REPEATS):
                with sw.checkpoint("mandelbrot"):
                    result = mb.mandelbrot(q, maxiter)

        assert result.shape == q.shape

    def test_median3x3_performance(self, backend):
        """Test median 3x3 kernel."""
        os.environ["YAPOCIS_BACKEND"] = backend

        median = Median()

        # Noisy image
        img = np.random.uniform(0, 255, (SMALL_SIZE, SMALL_SIZE)).astype(np.float32)

        with StopWatcher("median3x3", save_perf=True, kernel=median) as sw:
            for _ in range(REPEATS):
                with sw.checkpoint("median3x3"):
                    result = median.median3x3(img)

        assert result.shape == img.shape

    def test_gradient_performance(self, backend):
        """Test gradient kernels."""
        os.environ["YAPOCIS_BACKEND"] = backend

        grad = Gradient()

        # Test image
        img = np.random.uniform(0, 255, (SMALL_SIZE, SMALL_SIZE)).astype(np.float32)

        with StopWatcher("gradient", save_perf=True, kernel=grad) as sw:
            for _ in range(REPEATS):
                with sw.checkpoint("gradient"):
                    gx, gy = grad.gradient(img)

        assert gx.shape == img.shape
        assert gy.shape == img.shape

    def test_gaussian_performance(self, backend):
        """Test Gaussian blur at different scales."""
        os.environ["YAPOCIS_BACKEND"] = backend

        gaussian = Gaussian()

        # Test image
        img = np.random.uniform(0, 255, (SMALL_SIZE, SMALL_SIZE)).astype(np.float32)

        with StopWatcher("gaussian", save_perf=True, kernel=gaussian) as sw:
            num_scales = min(3, gaussian.get_scales())  # Test first 3 scales
            for scale in range(num_scales):
                for _ in range(REPEATS):
                    with sw.checkpoint(f"scale_{scale}"):
                        result = gaussian.gauss_image(img, scale)

        assert result.shape == img.shape

    def test_hsi_performance(self, backend):
        """Test HSI color space conversion kernels."""
        os.environ["YAPOCIS_BACKEND"] = backend

        hsi = HSI()

        # RGB image
        h, w = SMALL_SIZE, SMALL_SIZE
        rgb = np.random.uniform(0, 1, (h, w, 3)).astype(np.float32)

        with StopWatcher("hsi", save_perf=True, kernel=hsi) as sw:
            for _ in range(REPEATS):
                with sw.checkpoint("rgb2hsi3"):
                    hsi_img = hsi.rgb2hsi3(rgb)

                with sw.checkpoint("hsi2rgb3"):
                    rgb_back = hsi.hsi2rgb3(hsi_img)

        assert hsi_img.shape == rgb.shape

    def test_anisodiff_performance(self, backend):
        """Test anisotropic diffusion kernel."""
        os.environ["YAPOCIS_BACKEND"] = backend

        ad = Anisodiff()

        # Test image
        img = np.random.uniform(0, 255, (SMALL_SIZE, SMALL_SIZE)).astype(np.float32)

        with StopWatcher("anisodiff", save_perf=True, kernel=ad) as sw:
            for _ in range(REPEATS):
                with sw.checkpoint("anisodiff_5iter"):
                    result = ad.anisodiff_n(img, n=5, scale=3.0, step_size=0.2)

        assert result.shape == img.shape

    def test_truetrace_performance(self, backend):
        """Test truetrace segment extraction from binary mask."""
        os.environ["YAPOCIS_BACKEND"] = backend

        tt = TrueTrace()

        # Create a binary mask with a circle
        y, x = np.ogrid[:SMALL_SIZE, :SMALL_SIZE]
        center = SMALL_SIZE // 2
        radius = SMALL_SIZE // 4
        mask = ((x - center)**2 + (y - center)**2 <= radius**2).astype(np.uint8)

        with StopWatcher("truetrace", save_perf=True, kernel=tt) as sw:
            for _ in range(REPEATS):
                with sw.checkpoint("trace_segments"):
                    segs, (w, h) = tt.trace_segments(mask, iso=0.0)

        assert segs.shape[1] == 4  # Segments are Nx4

    def test_zcs_performance(self, backend):
        """Test zero-crossing detection kernel."""
        os.environ["YAPOCIS_BACKEND"] = backend

        zc = ZC()

        # Test image with gradients
        img = np.random.uniform(-10, 10, (SMALL_SIZE, SMALL_SIZE)).astype(np.float32)

        with StopWatcher("zcs", save_perf=True, kernel=zc) as sw:
            for _ in range(REPEATS):
                with sw.checkpoint("zcs"):
                    result = zc.zcs(img, threshold=0.1)

        assert result.shape == img.shape

    def test_graphics4_performance(self, backend):
        """Test graphics4 kernels with RGBA."""
        os.environ["YAPOCIS_BACKEND"] = backend

        g4 = Graphics4()

        # RGBA canvas
        h, w = SMALL_SIZE, SMALL_SIZE
        canvas = np.zeros((h, w, 4), dtype=np.float32)

        # Circles as list of tuples
        circles = [(50, 50, 20), (150, 150, 30), (100, 100, 15)]
        colors = np.array([
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0]
        ], dtype=np.float32)

        with StopWatcher("graphics4", save_perf=True, kernel=g4) as sw:
            for _ in range(REPEATS):
                with sw.checkpoint("circles"):
                    g4.circles(canvas, circles, colors=colors)

                with sw.checkpoint("read"):
                    result = g4.read(canvas)

        assert result.shape == canvas.shape


@pytest.mark.parametrize("backend", get_backends())
class TestLibraryPerformance:
    """Test high-level library API performance with realistic workflows."""

    def test_gaussian_multiscale_workflow(self, backend):
        """Test Gaussian blur at multiple scales."""
        os.environ["YAPOCIS_BACKEND"] = backend

        gaussian = Gaussian()
        img = np.random.uniform(0, 255, (MEDIUM_SIZE, MEDIUM_SIZE)).astype(np.float32)

        with StopWatcher("gaussian_multiscale", save_perf=True, kernel=gaussian) as sw:
            num_scales = min(5, gaussian.get_scales())
            for scale in range(num_scales):
                with sw.checkpoint(f"scale_{scale}"):
                    _ = gaussian.gauss_image(img, scale)

    def test_edge_detection_workflow(self, backend):
        """Test complete edge detection pipeline."""
        os.environ["YAPOCIS_BACKEND"] = backend

        gaussian = Gaussian()
        grad = Gradient()

        img = np.random.uniform(0, 255, (MEDIUM_SIZE, MEDIUM_SIZE)).astype(np.float32)

        with StopWatcher("edge_detection", save_perf=True) as sw:
            with sw.checkpoint("denoise"):
                smoothed = gaussian.gauss_image(img, scale=1)

            with sw.checkpoint("gradient"):
                gx, gy = grad.gradient(smoothed)

            with sw.checkpoint("magnitude"):
                magnitude = np.sqrt(gx**2 + gy**2)

    def test_color_conversion_workflow(self, backend):
        """Test HSI color space conversion pipeline."""
        os.environ["YAPOCIS_BACKEND"] = backend

        hsi = HSI()

        h, w = MEDIUM_SIZE, MEDIUM_SIZE
        rgb = np.random.uniform(0, 1, (h, w, 3)).astype(np.float32)

        with StopWatcher("color_conversion", save_perf=True, kernel=hsi) as sw:
            with sw.checkpoint("rgb2hsi3"):
                hsi_img = hsi.rgb2hsi3(rgb)

            with sw.checkpoint("hsi2rgb3"):
                rgb_back = hsi.hsi2rgb3(hsi_img)

    def test_graphics_rendering_workflow(self, backend):
        """Test complete graphics rendering workflow."""
        os.environ["YAPOCIS_BACKEND"] = backend

        g4 = Graphics4()

        h, w = MEDIUM_SIZE, MEDIUM_SIZE
        canvas = np.zeros((h, w, 4), dtype=np.float32)

        # Generate random shapes
        num_circles = 50

        circles = [(np.random.randint(0, w), np.random.randint(0, h), np.random.randint(5, 20))
                   for _ in range(num_circles)]
        colors = np.random.uniform(0, 1, (num_circles, 4)).astype(np.float32)
        colors[:, 3] = 1.0  # Full alpha

        with StopWatcher("graphics_rendering", save_perf=True, kernel=g4) as sw:
            with sw.checkpoint("draw_circles"):
                g4.circles(canvas, circles, colors=colors)

            with sw.checkpoint("read_canvas"):
                result = g4.read(canvas)

    def test_denoise_workflow(self, backend):
        """Test denoising workflow with multiple methods."""
        os.environ["YAPOCIS_BACKEND"] = backend

        gaussian = Gaussian()
        median = Median()
        ad = Anisodiff()

        # Noisy image
        img = np.random.uniform(0, 255, (MEDIUM_SIZE, MEDIUM_SIZE)).astype(np.float32)
        noise = np.random.normal(0, 20, img.shape).astype(np.float32)
        noisy = np.clip(img + noise, 0, 255).astype(np.float32)

        with StopWatcher("denoise", save_perf=True) as sw:
            with sw.checkpoint("gaussian_denoise"):
                _ = gaussian.gauss_image(noisy, scale=2)

            with sw.checkpoint("median_denoise"):
                _ = median.median3x3(noisy)

            with sw.checkpoint("anisodiff_denoise"):
                _ = ad.anisodiff_n(noisy, n=5, scale=3.0, step_size=0.2)



if __name__ == "__main__":
    # Run tests with performance tracking enabled
    os.environ["YAPOCIS_PERFORMANCE"] = "1"
    pytest.main([__file__, "-v", "--tb=short"])
