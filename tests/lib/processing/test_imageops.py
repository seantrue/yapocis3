#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.


import matplotlib.pyplot as plt
import numpy as np
import pytest

from yapocis.core import get_test_backends
from yapocis.lib.hsi import HSI
from yapocis.lib.imageops import Imageops
from yapocis.utils.basic import get_test_image

from . import StopWatcher


@pytest.mark.smoke
@pytest.mark.parametrize("backend", get_test_backends())
def test_clamp(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    imageops = Imageops()
    a = np.random.random((1000, 1000)).astype(np.float32)
    assert a.min() < 0.1 and a.max() > 0.9
    with StopWatcher("clamp", kernel=imageops) as sw:
        with sw.checkpoint("clamp_pixels"):
            b = imageops.clamp_pixels(a, 0.1, 0.9)
        sw.showcheck(b)
    assert b.min() >= 0.1 and b.max() <= 0.9


@pytest.mark.parametrize("backend", get_test_backends())
def test_binarize(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    imageops = Imageops()
    a = np.random.random((1000, 1000)).astype(np.float32)
    with StopWatcher("binarize", kernel=imageops) as sw:
        with sw.checkpoint("binarize"):
            b = imageops.binarize(a, 0.5)
        sw.showcheck(b)
    assert 450000 < b.sum() < 550000


@pytest.mark.parametrize("backend", get_test_backends())
def test_stretch(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    imageops = Imageops()
    a = np.random.uniform(
        0.1,
        0.9,
        (1000, 1000),
    ).astype(np.float32)
    assert a.min() >= 0.1 and a.max() <= 0.9
    with StopWatcher("stretch", kernel=imageops) as sw:
        with sw.checkpoint("stretch"):
            b = imageops.stretch(a, a.min(), a.max(), 0.0, 1.0)
        sw.showcheck(b)
    assert b.min() == 0.0 and b.max() >= 0.999999


@pytest.mark.parametrize("backend", get_test_backends())
def test_histogram(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    imageops = Imageops(FASTBINS=0)
    a = np.random.uniform(
        0.1,
        0.9,
        (1000, 1000),
    ).astype(np.float32)
    with StopWatcher("histogram", kernel=imageops) as sw:
        with sw.checkpoint("histogram_256"):
            h = imageops.histogram(a, n_bins=256)
            plt.plot(h)
            plt.show()
        with sw.checkpoint("histogram_1500"):
            h1 = imageops.histogram(a, n_bins=1500)
        plt.plot(h1)
        plt.show()
    assert h.sum() == a.shape[0] * a.shape[1]
    assert h.sum() == h1.sum()


@pytest.mark.parametrize("backend", get_test_backends())
def test_histogram_res(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    imageops = Imageops()
    a = np.random.uniform(
        0.1,
        0.9,
        (1000, 1000),
    ).astype(np.float32)
    with StopWatcher("histogram_res", kernel=imageops) as sw:
        with sw.checkpoint("histogram"):
            h = imageops.histogram(a, n_bins=256)
        assert h.sum() == a.shape[0] * a.shape[1]
        with sw.checkpoint("histogram_res"):
            imageops.histogram_res(a, h)
        with sw.checkpoint("read"):
            imageops.read(h)
    assert h.sum() == a.shape[0] * a.shape[1] * 2


@pytest.mark.parametrize("backend", get_test_backends())
def test_ops(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    NBINS = 1000
    imageops = Imageops(FASTBINS=0)
    a = np.random.normal(
        0.5,
        0.1,
        (1000, 1000),
    ).astype(np.float32)
    with StopWatcher("ops", kernel=imageops) as sw:
        with sw.checkpoint("clamp"):
            b = imageops.clamp_pixels(a, 0, 1 - 1e-05)
        sw.showcheck(b)
        with sw.checkpoint("histogram_full"):
            h = imageops.histogram(a, n_bins=NBINS)
        x = np.arange(0, 1, 1 / NBINS)
        plt.plot(x, h, label="full")
        with sw.checkpoint("stretch"):
            a = imageops.stretch(a, 0.15, 0.85, 0, 1)
        sw.showcheck(a)
        with sw.checkpoint("histogram_stretch"):
            h = imageops.histogram(a, n_bins=NBINS)
        plt.plot(x, h, label="stretch")
    plt.legend()
    plt.show()


@pytest.mark.parametrize("backend", get_test_backends())
def test_histeq_basic(backend, monkeypatch):
    """Test that histeq returns correct shape and dtype."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    imageops = Imageops()
    a = np.random.random((500, 500)).astype(np.float32)
    with StopWatcher("histeq_basic", kernel=imageops) as sw:
        with sw.checkpoint("histeq"):
            result = imageops.histeq(a)
        sw.showcheck(result)
    assert result.shape == a.shape
    assert result.dtype == np.float32


@pytest.mark.parametrize("backend", get_test_backends())
def test_histeq_spreads_histogram(backend, monkeypatch):
    """Test that histeq spreads out a narrow distribution."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    imageops = Imageops()
    # Create image with narrow distribution around 0.5
    a = np.random.normal(0.5, 0.05, (500, 500)).astype(np.float32)
    a = np.clip(a, 0.0, 1.0)
    original_std = a.std()

    with StopWatcher("histeq_spread", kernel=imageops) as sw:
        with sw.checkpoint("histeq"):
            result = imageops.histeq(a)
        sw.showcheck(result)

    # Histogram equalization should spread the distribution
    result_std = result.std()
    assert result_std > original_std * 1.5, "histeq should spread the distribution"


@pytest.mark.parametrize("backend", get_test_backends())
def test_histeq_output_range(backend, monkeypatch):
    """Test that histeq output is in valid range."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    imageops = Imageops()
    a = np.random.random((500, 500)).astype(np.float32)
    with StopWatcher("histeq_range", kernel=imageops) as sw:
        with sw.checkpoint("histeq"):
            result = imageops.histeq(a)
        sw.showcheck(result)
    assert result.min() >= 0.0, "histeq output should be >= 0"
    assert result.max() <= 1.0, "histeq output should be <= 1"


@pytest.mark.parametrize("backend", get_test_backends())
def test_histeq_hsi_roundtrip(backend, monkeypatch):
    """Test histogram equalization on I and S channels of an HSI image."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    # Get test image and convert to HSI
    rgb = get_test_image()
    hsi_converter = HSI()
    imageops = Imageops()

    with StopWatcher("histeq_hsi", kernel=imageops) as sw:
        # Convert RGB to HSI
        with sw.checkpoint("rgb2hsi"):
            hsi = hsi_converter.rgb2hsi3(rgb)

        # Extract H, S, I channels (last axis is channel)
        h = hsi[:, :, 0]
        s = hsi[:, :, 1]
        i = hsi[:, :, 2]

        # Equalize I (intensity) and S (saturation) channels
        with sw.checkpoint("histeq_intensity"):
            i_eq = imageops.histeq(i)
        with sw.checkpoint("histeq_saturation"):
            s_eq = imageops.histeq(s)

        # Reconstruct HSI with equalized channels
        hsi_eq = np.stack([h, s_eq, i_eq], axis=-1).astype(np.float32)

        # Convert back to RGB
        with sw.checkpoint("hsi2rgb"):
            rgb_eq = hsi_converter.hsi2rgb3(hsi_eq)

        sw.showcheck(rgb_eq)

    # Verify output shape and dtype
    assert rgb_eq.shape == rgb.shape, "Output shape should match input"
    assert rgb_eq.dtype == np.float32, "Output dtype should be float32"

    # Verify output is in valid range (may have small overflow due to color space conversion)
    assert rgb_eq.min() >= -0.01, "RGB output should be >= 0 (with tolerance)"
    assert rgb_eq.max() <= 1.01, "RGB output should be <= 1 (with tolerance)"


@pytest.mark.parametrize("backend", get_test_backends())
def test_histmatch_basic(backend, monkeypatch):
    """Test histogram matching returns correct shape and dtype."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    imageops = Imageops()

    # Create source image with narrow distribution
    src = np.random.normal(0.3, 0.1, (500, 500)).astype(np.float32)
    src = np.clip(src, 0.0, 1.0)

    # Create target histogram (uniform distribution)
    nbr_bins = 256
    target_hist = np.ones(nbr_bins, dtype=np.float32)

    with StopWatcher("histmatch_basic", kernel=imageops) as sw:
        with sw.checkpoint("histmatch"):
            result = imageops.histmatch(src, target_hist)
        sw.showcheck(result)

    assert result.shape == src.shape, "Output shape should match input"
    assert result.dtype == np.float32, "Output dtype should be float32"


@pytest.mark.parametrize("backend", get_test_backends())
def test_histmatch_to_uniform(backend, monkeypatch):
    """Test that matching to uniform histogram spreads distribution like histeq."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    imageops = Imageops()

    # Create source image with narrow distribution
    src = np.random.normal(0.5, 0.05, (500, 500)).astype(np.float32)
    src = np.clip(src, 0.0, 1.0)
    original_std = src.std()

    # Target: uniform histogram
    nbr_bins = 1024
    target_hist = np.ones(nbr_bins, dtype=np.float32)

    with StopWatcher("histmatch_uniform", kernel=imageops) as sw:
        with sw.checkpoint("histmatch"):
            result = imageops.histmatch(src, target_hist, nbr_bins=nbr_bins)
        sw.showcheck(result)

    # Matching to uniform should spread the distribution
    result_std = result.std()
    assert result_std > original_std * 1.5, (
        "Matching to uniform should spread distribution"
    )


@pytest.mark.parametrize("backend", get_test_backends())
def test_histmatch_to_bimodal(backend, monkeypatch):
    """Test histogram matching to a bimodal target distribution."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    imageops = Imageops()

    # Create source image with uniform distribution
    src = get_test_image()
    # src = np.random.uniform(0.0, 1.0, (500, 500)).astype(np.float32)

    # Create bimodal target histogram (peaks at 0.25 and 0.75)
    nbr_bins = 256
    target_hist = np.zeros(nbr_bins, dtype=np.float32)
    # Add Gaussian peaks
    x = np.linspace(0, 1, nbr_bins)
    target_hist += np.exp(-((x - 0.25) ** 2) / 0.01) * 1000
    target_hist += np.exp(-((x - 0.75) ** 2) / 0.01) * 1000

    with StopWatcher("histmatch_bimodal", kernel=imageops) as sw:
        with sw.checkpoint("histmatch"):
            result = imageops.histmatch(src, target_hist)
        sw.showcheck(result)

        # Compute histogram of result
        with sw.checkpoint("histogram"):
            result_hist = imageops.histogram(result, n_bins=nbr_bins)

    # Check that result has peaks near 0.25 and 0.75
    peak1_region = result_hist[50:80].sum()  # Around 0.25
    peak2_region = result_hist[175:205].sum()  # Around 0.75
    middle_region = result_hist[110:145].sum()  # Around 0.5

    # Peaks should have more mass than middle
    assert peak1_region > middle_region, "Should have peak near 0.25"
    assert peak2_region > middle_region, "Should have peak near 0.75"


@pytest.mark.parametrize("backend", get_test_backends())
def test_histmatch_between_images(backend, monkeypatch):
    """Test matching one image's histogram to another image's histogram."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    imageops = Imageops()

    # Source: dark image
    src = np.random.uniform(0.0, 0.3, (500, 500)).astype(np.float32)

    # Reference: bright image
    ref = np.random.uniform(0.7, 1.0, (500, 500)).astype(np.float32)

    nbr_bins = 256

    with StopWatcher("histmatch_images", kernel=imageops) as sw:
        # Get histogram of reference image
        with sw.checkpoint("ref_histogram"):
            ref_hist = imageops.histogram(ref, n_bins=nbr_bins)

        # Match source to reference
        with sw.checkpoint("histmatch"):
            result = imageops.histmatch(src, ref_hist, nbr_bins=nbr_bins)
        sw.showcheck(result)

    # Result should be brighter (closer to reference)
    assert result.mean() > src.mean() + 0.2, "Matched image should be brighter"
    assert result.mean() > 0.5, "Matched image should have mean > 0.5"
