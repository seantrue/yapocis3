#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

import numpy as np
import pytest
from pytest import approx

from yapocis.core import get_test_backends
from yapocis.lib.channels import split_planes
from yapocis.lib.gaussian import Gaussian, GaussianWeights
from yapocis.lib.gradient import Gradient
from yapocis.lib.hsi import HSI
from . import StopWatcher, image_test


@pytest.mark.smoke
def test_weight_sum():
    "Make sure that the kernel sums back to 1"
    g = GaussianWeights.compute_gaussian_weight()
    assert g.sum() == approx(1.0)


def test_weights():
    mw = 100
    gws = GaussianWeights(max_width=mw)
    for g in gws.gws:
        assert len(g) < mw


@pytest.mark.parametrize("backend", get_test_backends())
def test_kernels(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    g = Gaussian()
    a = np.zeros((400, 401), dtype=np.float32)
    a[127, 127] = 1.0
    with StopWatcher("impulse", kernel=g) as sw:
        for scale in range(g.get_scales()):
            with sw.checkpoint(f"{scale=}"):
                b = g.gauss_image(a, scale)
            sw.showcheck(b)
            gsum = g.gaussian_weights.get_gaussian_weight(scale).sum()
            bsum = b.sum()
            assert bsum == approx(gsum)
            assert bsum == approx(1.0)


@pytest.mark.parametrize("backend", get_test_backends())
def test_image(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    g = Gaussian()
    a = np.zeros((400, 401), dtype=np.float32)
    a[200, :] = 1.0
    a[:, 200] = 1.0
    for xy in range(400):
        a[xy, xy] = 1.0
    with StopWatcher("gauss_image", kernel=g) as sw:
        for scale in range(g.get_scales()):
            with sw.checkpoint(f"{scale=}"):
                b = g.gauss_image(a, scale)
            sw.showcheck(b)


@pytest.mark.parametrize("backend", get_test_backends())
def test_zcs(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    g = Gaussian()
    image = np.zeros((512, 513), dtype=np.float32)
    image[255, 240:280] = 1.0

    with StopWatcher(kernel=g) as sw:
        for scale in range(g.get_scales() - 1):
            with sw.checkpoint(f"{scale=}"):
                smaller = g.gauss_image(image, scale)
                larger = g.gauss_image(image, scale + 1)
                dog = smaller - larger
                zca = g.zcs(dog)
            sw.showcheck(dog, "dog")
            sw.showcheck(zca, "zcs")


@pytest.mark.parametrize("backend", get_test_backends())
def test_multiscale_zcs(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    a = np.zeros((400, 401), dtype=np.float32)
    a[150:250, 150:250] = 1.0
    g = Gaussian()
    with StopWatcher(kernel=g) as sw:
        for scale in range(g.get_scales() - 1):
            with sw.checkpoint(f"{scale=}"):
                zca, grad, theta = g.zcsdog(a, scale, res=True)
            sw.showcheck(zca)


@pytest.mark.parametrize("backend", get_test_backends())
def test_gradient(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    img = image_test()
    hsi = HSI()
    g = Gradient()
    with StopWatcher(kernel=g) as gsw, StopWatcher(kernel=hsi) as hsw:
        with hsw.checkpoint("hsi"):
            _, _, I = hsi.rgb2hsi(*split_planes(img))
        hsw.showcheck(I, "I")
        with gsw.checkpoint("gradient"):
            gr, th = g.gradient(I, 3)
        gsw.showcheck(th, "theta")
        gsw.showcheck(gr, "grad")


@pytest.mark.parametrize("backend", get_test_backends())
def test_multiscale_zcs_test(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    hsis = HSI()
    img = image_test()
    h, s, i = split_planes(hsis.hsi2rgb3(img))
    g = Gaussian()
    subtotal = 0.0
    with StopWatcher(kernel=g) as sw:
        for scale in range(g.get_scales() - 1):
            with sw.checkpoint(f"{scale=}"):
                zca, grad, theta = g.zcsdog(i, scale, res=True)
            sw.showcheck(zca, "zcs")
            sw.showcheck(grad, "grad")
            sw.showcheck(theta, "theta")
