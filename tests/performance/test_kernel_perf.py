#!/usr/bin/env python
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.
"""
Comprehensive performance testing for all yapocis library entry points.

Tests every kernel and library function across all available backends and devices,
saving performance data to the DuckDB database for analysis.
"""
import numpy as np
import pytest

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

from . import StopWatcher, get_backend_device_combinations

# Test data sizes - keep small for fast performance testing
SMALL_SIZE = 256
MEDIUM_SIZE = 512
REPEATS = 3  # Number of times to repeat each operation for consistent timing


@pytest.mark.performance
@pytest.mark.parametrize("backend,device", get_backend_device_combinations())
class TestKernelPerformance:
    """Test raw kernel performance across all backends and devices."""

    def test_operators_performance(self, backend, device, monkeypatch):
        """Test all operator kernels: add, sub, mul, divop."""
        monkeypatch.setenv("YAPOCIS_BACKEND", backend)
        monkeypatch.setenv("YAPOCIS_DEVICE", device)
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

    def test_mandelbrot_performance(self, backend, device, monkeypatch):
        """Test mandelbrot kernel."""
        monkeypatch.setenv("YAPOCIS_BACKEND", backend)
        monkeypatch.setenv("YAPOCIS_DEVICE", device)
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

    def test_median3x3_performance(self, backend, device, monkeypatch):
        """Test median 3x3 kernel."""
        monkeypatch.setenv("YAPOCIS_BACKEND", backend)
        monkeypatch.setenv("YAPOCIS_DEVICE", device)
        median = Median()

        # Noisy image
        img = np.random.uniform(0, 255, (SMALL_SIZE, SMALL_SIZE)).astype(np.float32)

        with StopWatcher("median3x3", save_perf=True, kernel=median) as sw:
            for _ in range(REPEATS):
                with sw.checkpoint("median3x3"):
                    result = median.median3x3(img)

        assert result.shape == img.shape

    def test_gradient_performance(self, backend, device, monkeypatch):
        """Test gradient kernels."""
        monkeypatch.setenv("YAPOCIS_BACKEND", backend)
        monkeypatch.setenv("YAPOCIS_DEVICE", device)

        grad = Gradient()

        # Test image
        img = np.random.uniform(0, 255, (SMALL_SIZE, SMALL_SIZE)).astype(np.float32)

        with StopWatcher("gradient", save_perf=True, kernel=grad) as sw:
            for _ in range(REPEATS):
                with sw.checkpoint("gradient"):
                    gx, gy = grad.gradient(img)

        assert gx.shape == img.shape
        assert gy.shape == img.shape

    def test_gaussian_performance(self, backend, device, monkeypatch):
        """Test Gaussian blur at different scales."""
        monkeypatch.setenv("YAPOCIS_BACKEND", backend)
        monkeypatch.setenv("YAPOCIS_DEVICE", device)

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

    def test_hsi_performance(self, backend, device, monkeypatch):
        """Test HSI color space conversion kernels."""
        monkeypatch.setenv("YAPOCIS_BACKEND", backend)
        monkeypatch.setenv("YAPOCIS_DEVICE", device)

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

    def test_anisodiff_performance(self, backend, device, monkeypatch):
        """Test anisotropic diffusion kernel."""
        monkeypatch.setenv("YAPOCIS_BACKEND", backend)
        monkeypatch.setenv("YAPOCIS_DEVICE", device)

        ad = Anisodiff()

        # Test image
        img = np.random.uniform(0, 255, (SMALL_SIZE, SMALL_SIZE)).astype(np.float32)

        with StopWatcher("anisodiff", save_perf=True, kernel=ad) as sw:
            for _ in range(REPEATS):
                with sw.checkpoint("anisodiff_5iter"):
                    result = ad.anisodiff_n(img, n=5, scale=3.0, step_size=0.2)

        assert result.shape == img.shape

    def test_truetrace_performance(self, backend, device, monkeypatch):
        """Test truetrace segment extraction from binary mask."""
        monkeypatch.setenv("YAPOCIS_BACKEND", backend)
        monkeypatch.setenv("YAPOCIS_DEVICE", device)
        # repeats = 1 if backend.startswith("num") else REPEATS
        repeats = REPEATS
        tt = TrueTrace()

        # Create a binary mask with a circle
        y, x = np.ogrid[:SMALL_SIZE, :SMALL_SIZE]
        center = SMALL_SIZE // 2
        radius = SMALL_SIZE // 4
        mask = ((x - center) ** 2 + (y - center) ** 2 <= radius**2).astype(np.uint8)

        with StopWatcher("truetrace", save_perf=True, kernel=tt) as sw:
            for _ in range(repeats):
                with sw.checkpoint("trace_segments"):
                    segs, (w, h) = tt.trace_segments(mask, iso=0.0)

        assert segs.shape[1] == 4  # Segments are Nx4

    def test_zcs_performance(self, backend, device, monkeypatch):
        """Test zero-crossing detection kernel."""
        monkeypatch.setenv("YAPOCIS_BACKEND", backend)
        monkeypatch.setenv("YAPOCIS_DEVICE", device)

        zc = ZC()

        # Test image with gradients
        img = np.random.uniform(-10, 10, (SMALL_SIZE, SMALL_SIZE)).astype(np.float32)

        with StopWatcher("zcs", save_perf=True, kernel=zc) as sw:
            for _ in range(REPEATS):
                with sw.checkpoint("zcs"):
                    result = zc.zcs(img, threshold=0.1)

        assert result.shape == img.shape

    def test_graphics4_performance(self, backend, device, monkeypatch):
        """Test all graphics4 functions: shapes, icons, and rotated text."""
        monkeypatch.setenv("YAPOCIS_BACKEND", backend)
        monkeypatch.setenv("YAPOCIS_DEVICE", device)
        repeats = REPEATS
        # repeats = 1 if backend.startswith("num") else REPEATS

        from tests.random_data import (
            random_circles,
            random_colors_4ch,
            random_composites_4ch,
            random_ellipses,
            random_icons,
            random_lines,
            random_rects,
            random_texts,
            random_triangles,
        )

        g4 = Graphics4()

        # RGBA canvas
        h, w = SMALL_SIZE, SMALL_SIZE
        canvas = np.zeros((h, w, 4), dtype=np.float32)

        # Generate test data using random_data functions
        n_shapes = 10
        lines = random_lines(n=n_shapes)
        circles = random_circles(n=n_shapes)
        rects = random_rects(n=n_shapes)
        triangles = random_triangles(n=n_shapes)
        ellipses = random_ellipses(n=n_shapes)
        colors = random_colors_4ch(n=n_shapes)

        # Icons and composites
        try:
            icons = random_icons(n=5)
            icon_locations, _ = random_composites_4ch(n=5)
            has_icons = True
        except Exception:
            has_icons = False

        # Text with rotations
        text_locs, text_strs, text_fonts, text_sizes, text_rots, text_alphas = (
            random_texts(n=5)
        )

        with StopWatcher("graphics4", save_perf=True, kernel=g4) as sw:
            for _ in range(repeats):
                with sw.checkpoint("lines"):
                    g4.lines(canvas, lines, colors=colors, distance=1.0)

                with sw.checkpoint("circles"):
                    g4.circles(canvas, circles, colors=colors)

                with sw.checkpoint("rects"):
                    g4.rects(canvas, rects, colors=colors)

                with sw.checkpoint("triangles"):
                    g4.triangles(canvas, triangles, colors=colors)

                with sw.checkpoint("ellipses"):
                    g4.ellipses(canvas, ellipses, colors=colors)

                if has_icons:
                    with sw.checkpoint("composite_icons"):
                        g4.composite(canvas, icon_locations, icons)

                with sw.checkpoint("text_rotated"):
                    for i in range(min(5, len(text_locs))):
                        g4.draw_strings(
                            canvas,
                            [text_locs[i]],
                            [text_strs[i]],
                            font=text_fonts[i],
                            size=text_sizes[i],
                            rotation=text_rots[i],
                            alpha=text_alphas[i],
                        )

                with sw.checkpoint("read"):
                    result = g4.read(canvas)

        assert result.shape == canvas.shape

    def test_simplex_performance(self, backend, device, monkeypatch):
        """Test simplex noise generation performance (2D and 3D)."""
        monkeypatch.setenv("YAPOCIS_BACKEND", backend)
        monkeypatch.setenv("YAPOCIS_DEVICE", device)

        from yapocis.lib.simplex import Simplex

        s = Simplex()

        with StopWatcher("simplex", save_perf=True, kernel=s) as sw:
            # 2D noise - various sizes
            for size in [256, 512]:
                for _ in range(REPEATS):
                    with sw.checkpoint(f"simplex2d_{size}x{size}"):
                        noise2d = s.simplex2d(
                            size, size, seed=42, octaves=4, scale=0.01
                        )

            # 2D noise - multiple octaves
            for octaves in [1, 4, 8]:
                with sw.checkpoint(f"simplex2d_octaves{octaves}"):
                    noise2d = s.simplex2d(
                        MEDIUM_SIZE, MEDIUM_SIZE, seed=42, octaves=octaves, scale=0.01
                    )

            # 2D tileable
            for _ in range(REPEATS):
                with sw.checkpoint("simplex2d_tileable"):
                    tileable = s.simplex2d(
                        SMALL_SIZE,
                        SMALL_SIZE,
                        seed=42,
                        octaves=4,
                        scale=0.01,
                        tileable=True,
                    )

            # 3D noise
            for depth in [32, 64]:
                with sw.checkpoint(f"simplex3d_{SMALL_SIZE}x{SMALL_SIZE}x{depth}"):
                    noise3d = s.simplex3d(
                        SMALL_SIZE, SMALL_SIZE, depth, seed=99, octaves=3, scale=0.02
                    )

        assert noise2d.shape == (MEDIUM_SIZE, MEDIUM_SIZE)
        assert noise3d.shape == (depth, SMALL_SIZE, SMALL_SIZE)
