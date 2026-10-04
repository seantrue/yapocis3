"""Tests verifying that hsi2rgb output stays within [0,1].

Although hsi2rgb lacks an explicit output clamp (unlike rgb2hsi which clamps to
[0, 1-eps]), the GLSL-style formula naturally bounds output to [0,1] for valid
inputs because: c = I * ((1-S) + S * clamp(p-1, 0, 1)), where the inner
expression is in [(1-S), 1] and I is in [0,1].

These tests confirm that property holds across backends and stress cases.
"""

import numpy as np
import pytest

from yapocis.core import get_test_backends
from yapocis.lib.hsi import HSI
from yapocis.utils.basic import get_test_image

from . import StopWatcher


@pytest.mark.parametrize("backend", get_test_backends())
def test_hsi2rgb_output_bounded(backend, monkeypatch):
    """Verify hsi2rgb output stays in [0,1] for valid HSI input across stress cases."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    hsi_converter = HSI()

    # Stress the conversion with high saturation and full hue/intensity sweep
    H, W = 200, 200
    h_channel = np.linspace(0, 1, H * W, dtype=np.float32).reshape(H, W)
    s_channel = np.full((H, W), 0.99, dtype=np.float32)
    i_channel = np.linspace(0.0, 1.0, H * W, dtype=np.float32).reshape(H, W)

    hsi = np.stack([h_channel, s_channel, i_channel], axis=-1).astype(np.float32)

    with StopWatcher("hsi2rgb_bounded", kernel=hsi_converter) as sw:
        with sw.checkpoint("hsi2rgb"):
            rgb = hsi_converter.hsi2rgb3(hsi)

    rgb_min = float(rgb.min())
    rgb_max = float(rgb.max())
    print(f"\n[{backend}] hsi2rgb output range: [{rgb_min:.6f}, {rgb_max:.6f}]")

    assert rgb_min >= 0.0, f"Output min {rgb_min} is below 0"
    assert rgb_max <= 1.0, f"Output max {rgb_max} exceeds 1"


@pytest.mark.parametrize("backend", get_test_backends())
def test_hsi2rgb_roundtrip_bounded(backend, monkeypatch):
    """RGB->HSI with boosted saturation->RGB stays in [0,1]."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    hsi_converter = HSI()

    img = get_test_image()
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    if img.ndim == 3 and img.shape[-1] > 3:
        img = img[..., :3]

    with StopWatcher("hsi2rgb_roundtrip", kernel=hsi_converter) as sw:
        with sw.checkpoint("rgb2hsi"):
            hsi = hsi_converter.rgb2hsi3(img)

        # Boost saturation to stress the inverse conversion
        hsi[..., 1] = np.clip(hsi[..., 1] * 1.5, 0.0, 1.0).astype(np.float32)

        with sw.checkpoint("hsi2rgb"):
            rgb = hsi_converter.hsi2rgb3(hsi)

    rgb_min = float(rgb.min())
    rgb_max = float(rgb.max())
    print(f"\n[{backend}] roundtrip output range: [{rgb_min:.6f}, {rgb_max:.6f}]")

    assert rgb_min >= 0.0, f"Output min {rgb_min} is below 0"
    assert rgb_max <= 1.0, f"Output max {rgb_max} exceeds 1"


@pytest.mark.parametrize("backend", get_test_backends())
def test_hsi2rgb_roundtrip_accuracy(backend, monkeypatch):
    """RGB->HSI->RGB roundtrip should be accurate (no saturation boost)."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    hsi_converter = HSI()

    img = get_test_image()
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    if img.ndim == 3 and img.shape[-1] > 3:
        img = img[..., :3]

    with StopWatcher("hsi2rgb_accuracy", kernel=hsi_converter) as sw:
        with sw.checkpoint("rgb2hsi"):
            hsi = hsi_converter.rgb2hsi3(img)
        with sw.checkpoint("hsi2rgb"):
            rgb = hsi_converter.hsi2rgb3(hsi)

    max_err = float(np.abs(rgb - img).max())
    mean_err = float(np.abs(rgb - img).mean())
    print(
        f"\n[{backend}] roundtrip max error: {max_err:.6f}, mean error: {mean_err:.6f}"
    )

    # Roundtrip should be accurate within floating point tolerance
    # The rgb2hsi clamp to 1-eps introduces small errors
    assert max_err < 0.02, f"Roundtrip max error {max_err} exceeds tolerance"
    assert mean_err < 0.005, f"Roundtrip mean error {mean_err} exceeds tolerance"
