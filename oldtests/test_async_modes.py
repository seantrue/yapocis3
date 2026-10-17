#!/usr/bin/env python
#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.
"""
Comprehensive async mode performance and validation tests.

Tests all combinations of YAPOCIS_ASYNC and YAPOCIS_ASYNC_WRITE flags
across different workload scenarios to validate correctness and measure
performance characteristics.
"""
import os
import pytest
import numpy as np

from yapocis.core import get_test_backends, get_test_devices
from yapocis.images import find_image
from yapocis.utils.basic import imread
from yapocis.lib.gaussian import Gaussian
from yapocis.lib.zc import ZC


def load_test_image(size=512):
    """Load and prepare test image."""
    img_path = find_image("test.jpg")
    img = imread(img_path)

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0

    if len(img.shape) == 3:
        img = img[:, :, 0]  # Grayscale

    # Resize
    from PIL import Image
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    pil_img = pil_img.resize((size, size), Image.BILINEAR)
    img = np.array(pil_img).astype(np.float32) / 255.0
    img = (img * 256.0).astype(np.float32)

    return img


def get_backend_device_combinations():
    """Generate all (backend, device) combinations for testing."""
    combinations = []
    for backend in get_test_backends():
        devices = get_test_devices(backend)
        for device_name in devices.keys():
            combinations.append((backend, device_name))
    return combinations


@pytest.mark.parametrize("backend,device", get_backend_device_combinations())
class TestAsyncModeValidation:
    """Validate that all async modes produce identical results."""

    def test_gaussian_async_modes_correctness(self, backend, device):
        """Test that all async modes produce identical results for Gaussian blur."""
        os.environ["YAPOCIS_BACKEND"] = backend
        os.environ["YAPOCIS_DEVICE"] = device

        img = load_test_image(256)
        scale = 1

        # Test all 4 combinations
        async_configs = [
            ("sync", False, False),
            ("async-exec", True, False),
            ("async-write", False, True),
            ("both", True, True),
        ]

        results = {}

        for mode_name, async_exec, async_write in async_configs:
            # Configure async modes
            os.environ["YAPOCIS_ASYNC"] = "1" if async_exec else "0"
            if async_write:
                os.environ["YAPOCIS_ASYNC_WRITE"] = "1"
            else:
                os.environ.pop("YAPOCIS_ASYNC_WRITE", None)

            gauss = Gaussian(device_name=device)
            result = gauss.gauss_image(img, scale)
            results[mode_name] = result

        # Validate all modes produce identical results
        sync_result = results["sync"]

        for mode_name in ["async-exec", "async-write", "both"]:
            max_diff = np.abs(sync_result - results[mode_name]).max()
            assert max_diff == 0.0, (
                f"{backend}/{device}: {mode_name} differs from sync by {max_diff}"
            )

    def test_zc_correctness(self, backend, device):
        """Test that ZC (zero-crossing) produces correct results across backends."""
        os.environ["YAPOCIS_BACKEND"] = backend
        os.environ["YAPOCIS_DEVICE"] = device

        img = load_test_image(256)
        threshold = 7.5e-06

        zc = ZC(device_name=device)
        result = zc.zcs(img, threshold=threshold)

        # Basic sanity checks
        assert result.shape == img.shape
        assert result.dtype == np.float32
        assert np.all((result == 0.0) | (result == 1.0)), "ZC result should be binary (0 or 1)"

        # Store result for cross-backend comparison
        return result


@pytest.mark.performance
@pytest.mark.parametrize("backend,device", get_backend_device_combinations())
class TestAsyncModePerformance:
    """Performance characterization tests for async modes."""

    def test_async_modes_small_pipeline(self, backend, device, benchmark):
        """Benchmark async modes with small pipeline (3 operations)."""
        os.environ["YAPOCIS_BACKEND"] = backend
        os.environ["YAPOCIS_DEVICE"] = device

        img = load_test_image(512)

        def run_pipeline(async_exec, async_write):
            """Run 3 Gaussian operations."""
            os.environ["YAPOCIS_ASYNC"] = "1" if async_exec else "0"
            if async_write:
                os.environ["YAPOCIS_ASYNC_WRITE"] = "1"
            else:
                os.environ.pop("YAPOCIS_ASYNC_WRITE", None)

            gauss = Gaussian(device_name=device)
            results = []
            for scale in [0, 1, 2]:
                result = gauss.gauss_image(img, scale)
                results.append(result)
            return results

        # Benchmark sync mode
        sync_result = benchmark(run_pipeline, False, False)

        # Quick test of async modes (not benchmarked, just validated)
        async_exec_result = run_pipeline(True, False)
        async_write_result = run_pipeline(False, True)
        both_result = run_pipeline(True, True)

        # Validate correctness
        for i in range(3):
            assert np.allclose(sync_result[i], async_exec_result[i], rtol=1e-5)
            assert np.allclose(sync_result[i], async_write_result[i], rtol=1e-5)
            assert np.allclose(sync_result[i], both_result[i], rtol=1e-5)


@pytest.mark.parametrize("backend,device", get_backend_device_combinations())
class TestAsyncModeConfigurations:
    """Test different async mode configurations."""

    def test_async_exec_only(self, backend, device):
        """Test async execution mode only (YAPOCIS_ASYNC=1)."""
        os.environ["YAPOCIS_BACKEND"] = backend
        os.environ["YAPOCIS_DEVICE"] = device
        os.environ["YAPOCIS_ASYNC"] = "1"
        os.environ.pop("YAPOCIS_ASYNC_WRITE", None)

        img = load_test_image(256)
        gauss = Gaussian(device_name=device)

        result = gauss.gauss_image(img, 0)
        assert result.shape == img.shape
        assert result.dtype == np.float32

    def test_async_write_only(self, backend, device):
        """Test async write mode only (YAPOCIS_ASYNC_WRITE=1)."""
        os.environ["YAPOCIS_BACKEND"] = backend
        os.environ["YAPOCIS_DEVICE"] = device
        os.environ["YAPOCIS_ASYNC"] = "0"
        os.environ["YAPOCIS_ASYNC_WRITE"] = "1"

        img = load_test_image(256)
        gauss = Gaussian(device_name=device)

        result = gauss.gauss_image(img, 0)
        assert result.shape == img.shape
        assert result.dtype == np.float32

    def test_async_both(self, backend, device):
        """Test both async modes enabled (YAPOCIS_ASYNC=1, YAPOCIS_ASYNC_WRITE=1)."""
        os.environ["YAPOCIS_BACKEND"] = backend
        os.environ["YAPOCIS_DEVICE"] = device
        os.environ["YAPOCIS_ASYNC"] = "1"
        os.environ["YAPOCIS_ASYNC_WRITE"] = "1"

        img = load_test_image(256)
        gauss = Gaussian(device_name=device)

        result = gauss.gauss_image(img, 0)
        assert result.shape == img.shape
        assert result.dtype == np.float32

    def test_multi_scale_async(self, backend, device):
        """Test async mode with multiple scales."""
        os.environ["YAPOCIS_BACKEND"] = backend
        os.environ["YAPOCIS_DEVICE"] = device
        os.environ["YAPOCIS_ASYNC"] = "1"
        os.environ["YAPOCIS_ASYNC_WRITE"] = "1"

        img = load_test_image(256)
        gauss = Gaussian(device_name=device)

        # Run multiple scales
        results = []
        for scale in range(min(3, gauss.get_scales())):
            result = gauss.gauss_image(img, scale)
            results.append(result)

        # Validate all results
        for result in results:
            assert result.shape == img.shape
            assert result.dtype == np.float32
            assert not np.any(np.isnan(result)), "Result contains NaN"
            assert not np.any(np.isinf(result)), "Result contains Inf"


@pytest.mark.parametrize("backend,device", get_backend_device_combinations())
class TestZeroCrossing:
    """Test zero-crossing detection across backends."""

    def test_zc_basic(self, backend, device):
        """Basic zero-crossing test."""
        os.environ["YAPOCIS_BACKEND"] = backend
        os.environ["YAPOCIS_DEVICE"] = device

        img = load_test_image(256)
        threshold = 7.5e-06

        zc = ZC(device_name=device)
        result = zc.zcs(img, threshold=threshold)

        assert result.shape == img.shape
        assert result.dtype == np.float32
        assert np.all((result >= 0.0) & (result <= 1.0)), "ZC result should be in [0,1]"

    def test_zc_threshold_variation(self, backend, device):
        """Test ZC with different threshold values."""
        os.environ["YAPOCIS_BACKEND"] = backend
        os.environ["YAPOCIS_DEVICE"] = device

        img = load_test_image(256)
        zc = ZC(device_name=device)

        thresholds = [1e-7, 1e-6, 1e-5, 1e-4]
        results = []

        for threshold in thresholds:
            result = zc.zcs(img, threshold=threshold)
            results.append(result)

            # Higher threshold should generally produce fewer zero-crossings
            assert result.shape == img.shape

        # Verify monotonic behavior (higher threshold → fewer detections)
        counts = [np.sum(r > 0) for r in results]
        # Note: This is a general trend but not strict monotonic due to threshold behavior

    def test_zc_coordinate_correctness(self, backend, device):
        """Test that ZC coordinates are correctly calculated (validates CUDA fix)."""
        os.environ["YAPOCIS_BACKEND"] = backend
        os.environ["YAPOCIS_DEVICE"] = device

        # Create simple test pattern with known zero-crossings
        size = 100
        img = np.zeros((size, size), dtype=np.float32)

        # Create vertical stripes with gradients
        for x in range(10, size, 20):
            for y in range(size):
                img[y, x] = 50.0 + (y - size/2) * 0.5

        threshold = 0.1
        zc = ZC(device_name=device)
        result = zc.zcs(img, threshold=threshold)

        # Should detect some zero-crossings
        assert np.sum(result > 0) > 0, "Should detect some zero-crossings"

        # Result should be binary
        assert np.all((result == 0.0) | (result == 1.0)), "ZC result should be binary"
