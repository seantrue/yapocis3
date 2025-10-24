#!/usr/bin/env python3
"""Comprehensive test of YAPOCIS_KERNELPACK functionality."""

import os

import numpy as np


def test_env_var_control():
    """Test 1: Environment variable enables kernel packages."""
    os.environ['YAPOCIS_BACKEND'] = 'ocl'
    os.environ['YAPOCIS_KERNELPACK'] = '1'

    from yapocis.lib.gaussian import Gaussian

    g = Gaussian()
    assert g.use_package == True, "Should use kernel package"


def test_explicit_override():
    """Test 2: Explicit parameter overrides environment."""

    os.environ['YAPOCIS_KERNELPACK'] = '1'

    from yapocis.lib.gaussian import Gaussian

    g1 = Gaussian(use_package=False)
    assert g1.use_package == False, "Explicit False should override env var"

    del os.environ['YAPOCIS_KERNELPACK']
    g2 = Gaussian(use_package=True)
    assert g2.use_package == True, "Explicit True should work without env var"


def test_correctness():
    """Test 3: Results are identical."""

    from yapocis.lib.gaussian import Gaussian

    np.random.seed(42)
    image = np.random.rand(256, 256).astype(np.float32)

    g_traditional = Gaussian(use_package=False)
    g_compiled = Gaussian(use_package=True)

    result_traditional = g_traditional.gauss_image(image.copy(), scale=3)
    result_compiled = g_compiled.gauss_image(image.copy(), scale=3)

    max_diff = np.abs(result_traditional - result_compiled).max()
    mean_diff = np.abs(result_traditional - result_compiled).mean()
    assert max_diff < 1e-10, "Max results differ by more than 1e-10"
    assert mean_diff < 1e-10, "Mean Results differ by more than 1e-10"


def test_performance():
    """Test 4: Performance improvement."""

    import time
    from yapocis.lib.gaussian import Gaussian

    np.random.seed(42)
    image = np.random.rand(256, 256).astype(np.float32)

    # Traditional path
    g_traditional = Gaussian(use_package=False)
    for _ in range(3):  # Warmup
        _ = g_traditional.gauss_image(image.copy(), scale=2)

    start = time.time()
    for _ in range(50):
        _ = g_traditional.gauss_image(image.copy(), scale=2)
    time_traditional = time.time() - start

    # Compiled path
    g_compiled = Gaussian(use_package=True)
    for _ in range(3):  # Warmup
        _ = g_compiled.gauss_image(image.copy(), scale=2)

    start = time.time()
    for _ in range(50):
        _ = g_compiled.gauss_image(image.copy(), scale=2)
    time_compiled = time.time() - start

    speedup = time_traditional / time_compiled
    improvement = (speedup - 1) * 100

    assert speedup > 1.0, "Compiled should be faster"


def test_multiple_scales():
    """Test 5: Multiple Gaussian scales."""

    from yapocis.lib.gaussian import Gaussian

    g = Gaussian(use_package=True)
    image = np.random.rand(100, 100).astype(np.float32)

    num_scales = min(5, g.get_scales())
    print(f"Testing {num_scales} scales...")

    for scale in range(num_scales):
        result = g.gauss_image(image, scale=scale)
        kernel = g.get_gaussian_kernel(scale)
        print(f"  Scale {scale} (width={kernel.width}): "
              f"range=[{result.min():.4f}, {result.max():.4f}]")

        assert result.shape == image.shape
        assert not np.isnan(result).any()


def test_zcsdog():
    """Test 6: Complex operation (zcsdog)."""

    from yapocis.lib.gaussian import Gaussian

    g_traditional = Gaussian(use_package=False)
    g_compiled = Gaussian(use_package=True)

    np.random.seed(42)
    image = np.random.rand(200, 200).astype(np.float32)

    zca1, grad1, theta1 = g_traditional.zcsdog(image.copy(), scale=2)
    zca2, grad2, theta2 = g_compiled.zcsdog(image.copy(), scale=2)

    diff_zca = np.abs(zca1 - zca2).max()
    diff_grad = np.abs(grad1 - grad2).max()
    diff_theta = np.abs(theta1 - theta2).max()

    assert diff_zca < 1e-6, "ZCA should match"
    assert diff_grad < 1e-6, "Gradient should match"
    assert diff_theta < 1e-6, "Theta should match"
