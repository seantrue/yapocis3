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

from tests import StopWatcher
from yapocis.core import get_backends

# Import library classes that we'll test
from yapocis.lib.anisodiff import Anisodiff
from yapocis.lib.gaussian import Gaussian
from yapocis.lib.gradient import Gradient
from yapocis.lib.graphics4 import Graphics4
from yapocis.lib.hsi import HSI
from yapocis.lib.median import Median

# Test data sizes - keep small for fast performance testing
SMALL_SIZE = 256
MEDIUM_SIZE = 512
REPEATS = 3  # Number of times to repeat each operation for consistent timing


@pytest.mark.performance
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
        num_circles = 200

        circles = [
            (np.random.randint(0, w), np.random.randint(0, h), np.random.randint(5, 20))
            for _ in range(num_circles)
        ]
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
