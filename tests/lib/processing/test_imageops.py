#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.


import matplotlib.pyplot as plt
import numpy as np
import pytest

from yapocis.core import get_test_backends
from yapocis.lib.imageops import Imageops

from . import StopWatcher


@pytest.mark.smoke
@pytest.mark.parametrize("backend", get_test_backends())
def test_clamp(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    imageops = Imageops()
    a = np.random.random(1000).astype(np.float32)
    assert a.min() < 0.1 and a.max() > 0.9
    with StopWatcher("clamp", kernel=imageops) as sw:
        with sw.checkpoint("clamp_pixels"):
            b = imageops.clamp_pixels(a, 0.1, 0.9)
    assert b.min() >= 0.1 and b.max() <= 0.9


@pytest.mark.parametrize("backend", get_test_backends())
def test_binarize(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    imageops = Imageops()
    a = np.random.random(1000).astype(np.float32)
    with StopWatcher("binarize", kernel=imageops) as sw:
        with sw.checkpoint("binarize"):
            b = imageops.binarize(a, 0.5)
    assert 450 < b.sum() < 550


@pytest.mark.parametrize("backend", get_test_backends())
def test_stretch(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    imageops = Imageops()
    a = np.random.uniform(
        0.1,
        0.9,
        (2000, 2000),
    ).astype(np.float32)
    assert a.min() >= 0.1 and a.max() <= 0.9
    with StopWatcher("stretch", kernel=imageops) as sw:
        with sw.checkpoint("stretch"):
            b = imageops.stretch(a, a.min(), a.max(), 0.0, 1.0)
    assert b.min() == 0.0 and b.max() >= 0.999999


@pytest.mark.parametrize("backend", get_test_backends())
def test_histogram(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    imageops = Imageops(FASTBINS=0)
    a = np.random.uniform(
        0.1,
        0.9,
        (2000, 2000),
    ).astype(np.float32)
    with StopWatcher("histogram", kernel=imageops) as sw:
        with sw.checkpoint("histogram_256"):
            h = imageops.histogram(a, n_bins=256)
        with sw.checkpoint("histogram_1500"):
            h1 = imageops.histogram(a, n_bins=1500)
    assert h.sum() == a.shape[0] * a.shape[1]
    assert h.sum() == h1.sum()


@pytest.mark.parametrize("backend", get_test_backends())
def test_histogram_res(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    imageops = Imageops()
    a = np.random.uniform(
        0.1,
        0.9,
        (2000, 2000),
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
        (2000, 2000),
    ).astype(np.float32)
    with StopWatcher("ops", kernel=imageops) as sw:
        with sw.checkpoint("clamp"):
            b = imageops.clamp_pixels(a, 0, 1 - 1e-05)
        with sw.checkpoint("histogram_full"):
            h = imageops.histogram(a, n_bins=NBINS)
        x = np.arange(0, 1, 1 / NBINS)
        plt.plot(x, h, label="full")
        with sw.checkpoint("stretch"):
            a = imageops.stretch(a, 0.15, 0.85, 0, 1)
        with sw.checkpoint("histogram_stretch"):
            h = imageops.histogram(a, n_bins=NBINS)
        plt.plot(x, h, label="stretch")
    plt.legend()
    plt.show()
