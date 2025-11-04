#!/usr/bin/env python
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.
"""
Performance tests for async-enabled library functions.

Tests async vs sync execution modes for:
- Anisodiff (anisotropic diffusion)
- Gaussian (separable convolution)
- VGG16 and VGG16Resident (CNN inference)
- ResNet50Resident (CNN inference)

Uses test.jpg image asset and compares performance with YAPOCIS_ASYNC enabled/disabled.
"""
import os

import numpy as np
import pytest

from yapocis.core import get_test_backends, get_test_devices
from yapocis.images import find_image
from yapocis.utils.basic import imread
from yapocis.lib.anisodiff import Anisodiff
from yapocis.lib.gaussian import Gaussian
from . import StopWatcher

# Test configuration
REPEATS = 3  # Number of times to repeat each operation


def load_test_image(size=None):
    """Load test.jpg image from assets."""
    img_path = find_image("test.jpg")
    img = imread(img_path)

    # Convert to float32 and normalize to [0, 1]
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0

    # Resize if requested
    if size is not None:
        from PIL import Image
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        pil_img = pil_img.resize((size, size), Image.BILINEAR)
        img = np.array(pil_img).astype(np.float32) / 255.0

    return img


def get_backend_device_combinations():
    """Generate all (backend, device) combinations for testing."""
    combinations = []

    for backend in get_test_backends():
        devices = get_test_devices(backend)
        for device_name in devices.keys():
            combinations.append((backend, device_name))

    return combinations


@pytest.mark.performance
@pytest.mark.parametrize("backend,device", get_backend_device_combinations())
class TestAsyncPerformance:
    """Test async vs sync performance for library functions."""

    def test_anisodiff_async_vs_sync(self, backend, device):
        """Test anisotropic diffusion with async vs sync execution."""
        os.environ["YAPOCIS_BACKEND"] = backend
        os.environ["YAPOCIS_DEVICE"] = device

        # Load test image (grayscale)
        img = load_test_image(size=512)
        if len(img.shape) == 3:
            img = img[:, :, 0]  # Take first channel
        img = (img * 256.0).astype(np.float32)  # Scale to [0, 256]

        # Test parameters
        n_iterations = 10
        scale = 3.0
        step_size = 0.2

        # Test SYNC mode
        os.environ["YAPOCIS_ASYNC"] = "0"
        ad_sync = Anisodiff(device_name=device)

        with StopWatcher("anisodiff_sync", save_perf=True, kernel=ad_sync) as sw:
            for _ in range(REPEATS):
                with sw.checkpoint(f"anisodiff_{n_iterations}iter_sync"):
                    result_sync = ad_sync.anisodiff_n(img, n=n_iterations, scale=scale, step_size=step_size)

        # Test ASYNC mode
        os.environ["YAPOCIS_ASYNC"] = "1"
        ad_async = Anisodiff(device_name=device)

        with StopWatcher("anisodiff_async", save_perf=True, kernel=ad_async) as sw:
            for _ in range(REPEATS):
                with sw.checkpoint(f"anisodiff_{n_iterations}iter_async"):
                    result_async = ad_async.anisodiff_n(img, n=n_iterations, scale=scale, step_size=step_size)

        # Verify results are similar (async shouldn't change output)
        assert result_sync.shape == result_async.shape
        np.testing.assert_allclose(result_sync, result_async, rtol=1e-4, atol=1e-5)

    def test_gaussian_async_vs_sync(self, backend, device):
        """Test Gaussian blur with async vs sync execution."""
        os.environ["YAPOCIS_BACKEND"] = backend
        os.environ["YAPOCIS_DEVICE"] = device

        # Load test image (grayscale)
        img = load_test_image(size=512)
        if len(img.shape) == 3:
            img = img[:, :, 0]  # Take first channel
        img = (img * 256.0).astype(np.float32)  # Scale to [0, 256]

        # Test SYNC mode
        os.environ["YAPOCIS_ASYNC"] = "0"
        gauss_sync = Gaussian(device_name=device)

        with StopWatcher("gaussian_sync", save_perf=True, kernel=gauss_sync) as sw:
            # Test multiple scales
            num_scales = min(3, gauss_sync.get_scales())
            for scale in range(num_scales):
                for _ in range(REPEATS):
                    with sw.checkpoint(f"gauss_image_scale{scale}_sync"):
                        result_sync = gauss_sync.gauss_image(img, scale)

        # Test ASYNC mode
        os.environ["YAPOCIS_ASYNC"] = "1"
        gauss_async = Gaussian(device_name=device)

        with StopWatcher("gaussian_async", save_perf=True, kernel=gauss_async) as sw:
            # Test same scales
            num_scales = min(3, gauss_async.get_scales())
            for scale in range(num_scales):
                for _ in range(REPEATS):
                    with sw.checkpoint(f"gauss_image_scale{scale}_async"):
                        result_async = gauss_async.gauss_image(img, scale)

        # Verify results are similar
        assert result_sync.shape == result_async.shape
        np.testing.assert_allclose(result_sync, result_async, rtol=1e-4, atol=1e-5)

    def test_gaussian_zcsdog_async_vs_sync(self, backend, device):
        """Test Gaussian zero-crossing DOG with async vs sync execution."""
        os.environ["YAPOCIS_BACKEND"] = backend
        os.environ["YAPOCIS_DEVICE"] = device

        # Load test image (grayscale)
        img = load_test_image(size=512)
        if len(img.shape) == 3:
            img = img[:, :, 0]  # Take first channel
        img = (img * 256.0).astype(np.float32)  # Scale to [0, 256]

        scale = 1
        threshold = 7.5e-06

        # Test SYNC mode
        os.environ["YAPOCIS_ASYNC"] = "0"
        gauss_sync = Gaussian(device_name=device)

        with StopWatcher("gaussian_zcsdog_sync", save_perf=True, kernel=gauss_sync) as sw:
            for _ in range(REPEATS):
                with sw.checkpoint("zcsdog_sync"):
                    zca_sync, grad_sync, theta_sync = gauss_sync.zcsdog(
                        img, scale, threshold=threshold, res=True
                    )

        # Test ASYNC mode
        os.environ["YAPOCIS_ASYNC"] = "1"
        gauss_async = Gaussian(device_name=device)

        with StopWatcher("gaussian_zcsdog_async", save_perf=True, kernel=gauss_async) as sw:
            for _ in range(REPEATS):
                with sw.checkpoint("zcsdog_async"):
                    zca_async, grad_async, theta_async = gauss_async.zcsdog(
                        img, scale, threshold=threshold, res=True
                    )

        # Verify results are similar
        assert zca_sync.shape == zca_async.shape
        # Note: Small differences expected due to floating point ordering
        np.testing.assert_allclose(zca_sync, zca_async, rtol=1e-3, atol=1e-4)


@pytest.mark.performance
@pytest.mark.parametrize("backend,device", get_backend_device_combinations())
class TestCNNAsyncPerformance:
    """Test async performance for CNN inference (VGG16, ResNet50)."""

    def test_vgg16_async_vs_sync(self, backend, device):
        """Test VGG16 with async vs sync execution."""
        os.environ["YAPOCIS_BACKEND"] = backend
        os.environ["YAPOCIS_DEVICE"] = device

        # Skip if weights not available
        try:
            from yapocis.models import get_weights_path
            weights_path = get_weights_path('vgg16')
            if weights_path is None:
                pytest.skip("VGG16 weights not available")
        except Exception:
            pytest.skip("VGG16 weights not available")

        from yapocis.lib.vgg16 import VGG16

        # Load and preprocess test image
        img = load_test_image()

        # Test SYNC mode
        os.environ["YAPOCIS_ASYNC"] = "0"
        vgg_sync = VGG16(str(weights_path))
        preprocessed = vgg_sync.preprocess_image(img)

        with StopWatcher("vgg16_sync", save_perf=True, kernel=vgg_sync) as sw:
            for _ in range(REPEATS):
                with sw.checkpoint("forward_sync"):
                    result_sync = vgg_sync.forward(preprocessed)

        # Test ASYNC mode
        os.environ["YAPOCIS_ASYNC"] = "1"
        vgg_async = VGG16(str(weights_path))

        with StopWatcher("vgg16_async", save_perf=True, kernel=vgg_async) as sw:
            for _ in range(REPEATS):
                with sw.checkpoint("forward_async"):
                    result_async = vgg_async.forward(preprocessed)

        # Verify results are similar
        assert result_sync.shape == result_async.shape
        np.testing.assert_allclose(result_sync, result_async, rtol=1e-3, atol=1e-4)

    def test_vgg16_resident_async_vs_sync(self, backend, device):
        """Test VGG16Resident with async vs sync execution."""
        os.environ["YAPOCIS_BACKEND"] = backend
        os.environ["YAPOCIS_DEVICE"] = device

        # Skip if weights not available
        try:
            from yapocis.models import get_weights_path
            weights_path = get_weights_path('vgg16')
            if weights_path is None:
                pytest.skip("VGG16 weights not available")
        except Exception:
            pytest.skip("VGG16 weights not available")

        from yapocis.lib.vgg16 import VGG16Resident

        # Load and preprocess test image
        img = load_test_image()

        # Test SYNC mode
        os.environ["YAPOCIS_ASYNC"] = "0"
        vgg_sync = VGG16Resident(str(weights_path))
        preprocessed = vgg_sync.preprocess_image(img)

        with StopWatcher("vgg16_resident_sync", save_perf=True, kernel=vgg_sync) as sw:
            for _ in range(REPEATS):
                with sw.checkpoint("forward_resident_sync"):
                    result_sync = vgg_sync.forward(preprocessed)

        # Test ASYNC mode
        os.environ["YAPOCIS_ASYNC"] = "1"
        vgg_async = VGG16Resident(str(weights_path))

        with StopWatcher("vgg16_resident_async", save_perf=True, kernel=vgg_async) as sw:
            for _ in range(REPEATS):
                with sw.checkpoint("forward_resident_async"):
                    result_async = vgg_async.forward(preprocessed)

        # Verify results are similar
        assert result_sync.shape == result_async.shape
        np.testing.assert_allclose(result_sync, result_async, rtol=1e-3, atol=1e-4)

    def test_resnet50_resident_async_vs_sync(self, backend, device):
        """Test ResNet50Resident with async vs sync execution."""
        os.environ["YAPOCIS_BACKEND"] = backend
        os.environ["YAPOCIS_DEVICE"] = device

        # Skip if weights not available
        try:
            from yapocis.models import get_weights_path
            weights_path = get_weights_path('resnet50')
            if weights_path is None:
                pytest.skip("ResNet50 weights not available")
        except Exception:
            pytest.skip("ResNet50 weights not available")

        from yapocis.lib.resnet50 import ResNet50Resident

        # Load and preprocess test image
        img = load_test_image()

        # Test SYNC mode
        os.environ["YAPOCIS_ASYNC"] = "0"
        resnet_sync = ResNet50Resident(str(weights_path))
        preprocessed = resnet_sync.preprocess_image(img)

        with StopWatcher("resnet50_resident_sync", save_perf=True, kernel=resnet_sync) as sw:
            for _ in range(REPEATS):
                with sw.checkpoint("forward_resident_sync"):
                    result_sync = resnet_sync.forward(preprocessed)

        # Test ASYNC mode
        os.environ["YAPOCIS_ASYNC"] = "1"
        resnet_async = ResNet50Resident(str(weights_path))

        with StopWatcher("resnet50_resident_async", save_perf=True, kernel=resnet_async) as sw:
            for _ in range(REPEATS):
                with sw.checkpoint("forward_resident_async"):
                    result_async = resnet_async.forward(preprocessed)

        # Verify results are similar
        assert result_sync.shape == result_async.shape
        np.testing.assert_allclose(result_sync, result_async, rtol=1e-3, atol=1e-4)
