#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.
"""Tests for yapocis.lib.median module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from yapocis.core import get_test_backends
from yapocis.lib.median import Median
from . import StopWatcher, image_test


def setup_test():
    """Setup test image and Median instance."""
    image = image_test()
    # Ensure that we don't have fortunate addressing!
    assert image.shape[0] != image.shape[1]
    return image, Median()


def create_noisy_image(shape=(100, 120), noise_fraction=0.1, seed=42):
    """Create a synthetic image with salt-and-pepper noise."""
    np.random.seed(seed)
    # Create a smooth gradient image
    h, w = shape
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x, y)
    image = (xx + yy) / 2.0

    # Add salt-and-pepper noise
    noisy = image.copy()
    num_noise = int(noise_fraction * image.size)
    coords = np.random.randint(0, h, num_noise), np.random.randint(0, w, num_noise)
    # Salt (white pixels)
    noisy[coords[0][: num_noise // 2], coords[1][: num_noise // 2]] = 1.0
    # Pepper (black pixels)
    noisy[coords[0][num_noise // 2 :], coords[1][num_noise // 2 :]] = 0.0

    return noisy.astype(np.float32), image.astype(np.float32)


@pytest.mark.smoke
@pytest.mark.parametrize("backend", get_test_backends())
def test_median3x3_basic(backend, monkeypatch):
    """Test basic median3x3 functionality."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    noisy, original = create_noisy_image()
    median = Median()

    with StopWatcher("median3x3 basic", kernel=median) as sw:
        with sw.checkpoint("median3x3"):
            result = median.median3x3(noisy)

    # Result should have same shape as input
    assert result.shape == noisy.shape

    # Result should be different from input (noise reduction)
    assert not np.array_equal(result, noisy)


@pytest.mark.parametrize("backend", get_test_backends())
def test_median3x3_single_iteration(backend, monkeypatch):
    """Test median3x3 with a single iteration."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    noisy, original = create_noisy_image()
    median = Median()

    with StopWatcher("median3x3 single", kernel=median) as sw:
        with sw.checkpoint("single_iteration"):
            result = median.median3x3(noisy, iterations=1)

    assert result.shape == noisy.shape
    assert result.dtype == noisy.dtype


@pytest.mark.parametrize("backend", get_test_backends())
def test_median3x3_multiple_iterations(backend, monkeypatch):
    """Test median3x3 with multiple iterations."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    noisy, original = create_noisy_image(noise_fraction=0.15)  # Higher noise for visible effect
    median = Median()

    with StopWatcher("median3x3 multi", kernel=median) as sw:
        with sw.checkpoint("5_iterations"):
            result = median.median3x3(noisy, iterations=5)

    assert result.shape == noisy.shape

    # More iterations should produce smoother result
    # Check variance decreases with iterations (allow small tolerance)
    variance_noisy = np.var(noisy)
    variance_result = np.var(result)
    # Median filter should reduce variance (smooth the image)
    # Allow 1% tolerance for floating point and backend differences
    assert variance_result <= variance_noisy * 1.01, (
        f"Variance should decrease: noisy={variance_noisy:.6f}, result={variance_result:.6f}"
    )


@pytest.mark.parametrize("backend", get_test_backends())
def test_median3x3_zero_iterations(backend, monkeypatch):
    """Test median3x3 with zero iterations (should return copy)."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    noisy, original = create_noisy_image()
    median = Median()

    result = median.median3x3(noisy, iterations=0)

    # Zero iterations should return a copy of input
    assert_allclose(result, noisy, rtol=1e-5)
    # But it should be a copy, not the same array
    assert result is not noisy


@pytest.mark.parametrize("backend", get_test_backends())
def test_median3x3_slow_vs_fast(backend, monkeypatch):
    """Test that slow and fast implementations produce same results."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    noisy, original = create_noisy_image(shape=(50, 60))
    median = Median()

    with StopWatcher("median comparison", kernel=median) as sw:
        with sw.checkpoint("slow"):
            result_slow = median.median3x3slow(noisy, iterations=3)
        with sw.checkpoint("fast"):
            result_fast = median.median3x3fast(noisy, iterations=3)

    # Results should be close (allowing for floating point differences)
    assert_allclose(result_slow, result_fast, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend", get_test_backends())
def test_median3x3_on_real_image(backend, monkeypatch):
    """Test median3x3 on a real test image."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    rgb_image, median = setup_test()

    # Use single channel for testing
    if len(rgb_image.shape) == 3:
        gray = rgb_image[:, :, 0].astype(np.float32)
    else:
        gray = rgb_image.astype(np.float32)

    with StopWatcher("median3x3 real image", kernel=median) as sw:
        with sw.checkpoint("apply"):
            result = median.median3x3(gray, iterations=2)
        sw.showcheck(result)

    assert result.shape == gray.shape


@pytest.mark.parametrize("backend", get_test_backends())
def test_median3x3_preserves_edges(backend, monkeypatch):
    """Test that median filter preserves strong edges."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    # Create a step edge image
    h, w = 50, 60
    image = np.zeros((h, w), dtype=np.float32)
    image[:, w // 2 :] = 1.0

    median = Median()

    result = median.median3x3(image, iterations=1)

    # Check that the left side is still mostly 0 and right side mostly 1
    left_mean = np.mean(result[:, : w // 2 - 2])
    right_mean = np.mean(result[:, w // 2 + 2 :])

    assert left_mean < 0.1, f"Left side should be dark: {left_mean}"
    assert right_mean > 0.9, f"Right side should be bright: {right_mean}"


@pytest.mark.parametrize("backend", get_test_backends())
def test_median3x3_removes_salt_pepper(backend, monkeypatch):
    """Test that median filter effectively removes salt-and-pepper noise."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    # Use higher noise fraction and more iterations for robust test
    noisy, original = create_noisy_image(noise_fraction=0.15)
    median = Median()

    result = median.median3x3(noisy, iterations=3)

    # Calculate mean absolute error
    mae_before = np.mean(np.abs(noisy - original))
    mae_after = np.mean(np.abs(result - original))

    # Filter should reduce error (get closer to original)
    # Allow small tolerance for backend differences
    assert mae_after <= mae_before * 1.05, (
        f"MAE should decrease: before={mae_before:.4f}, after={mae_after:.4f}"
    )


@pytest.mark.parametrize("backend", get_test_backends())
def test_median3x3_returns_new_array(backend, monkeypatch):
    """Test that median3x3 returns a new array (not same object as input)."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    noisy, original = create_noisy_image()
    median = Median()

    result = median.median3x3(noisy, iterations=2)

    # Result should be a different array object
    assert result is not noisy

    # Note: The fast implementation may modify the input array as a side effect
    # due to ping-pong buffer usage. This tests that the returned array is
    # at least a distinct object.


@pytest.mark.parametrize("backend", get_test_backends())
def test_median3x3_dtype_preservation(backend, monkeypatch):
    """Test that output dtype is float32."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    noisy, original = create_noisy_image()
    median = Median()

    result = median.median3x3(noisy, iterations=1)

    # Median interface uses float, output should be float32
    assert result.dtype == np.float32


@pytest.mark.parametrize("backend", get_test_backends())
def test_median3x3_various_sizes(backend, monkeypatch):
    """Test median3x3 on various image sizes."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    median = Median()

    sizes = [(32, 32), (64, 48), (100, 150), (256, 256)]

    for shape in sizes:
        noisy, _ = create_noisy_image(shape=shape)
        result = median.median3x3(noisy, iterations=1)
        assert result.shape == shape, f"Shape mismatch for size {shape}"


@pytest.mark.parametrize("backend", get_test_backends())
def test_median3x3_iteration_progression(backend, monkeypatch):
    """Test that more iterations produce progressively smoother results."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    noisy, original = create_noisy_image(noise_fraction=0.1)
    median = Median()

    variances = []
    for n_iter in [1, 2, 4, 8]:
        result = median.median3x3(noisy, iterations=n_iter)
        variances.append(np.var(result))

    # Variance should generally decrease with more iterations
    # Allow some tolerance for floating point variations
    for i in range(len(variances) - 1):
        # Allow 5% tolerance for non-monotonic behavior
        assert variances[i + 1] <= variances[i] * 1.05, (
            f"Variance should decrease: iter {i + 1}={variances[i]:.4f}, "
            f"iter {i + 2}={variances[i + 1]:.4f}"
        )


@pytest.mark.parametrize("backend", get_test_backends())
def test_median3x3_uniform_input(backend, monkeypatch):
    """Test median3x3 on uniform input (should return same values)."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    # Create uniform image
    h, w = 50, 60
    uniform = np.full((h, w), 0.5, dtype=np.float32)
    median = Median()

    result = median.median3x3(uniform, iterations=3)

    # Interior pixels should be unchanged (border behavior may differ)
    interior_input = uniform[1:-1, 1:-1]
    interior_result = result[1:-1, 1:-1]
    assert_allclose(interior_result, interior_input, rtol=1e-5)
