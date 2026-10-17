#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.

import pytest

from yapocis.core import get_backend, get_test_backends
from yapocis.lib.anisodiff import Anisodiff
from yapocis.lib.channels import join_planes, split_planes
from yapocis.lib.hsi import HSI
from yapocis.utils.config import get_device_config
from yapocis.utils.stereo_typing import *

from . import StopWatcher, image_test

N = 100


def get_available_device_names():
    if get_backend() == "ocl":
        from yapocis.ocl.device import get_available_device_names as gadn

        return gadn()
    return [""]


def setup_test(device_name: str | None = None):
    image = image_test()
    # Ensure that we don't have fortunate addressing!
    assert image.shape[0] != image.shape[1]
    return image, Anisodiff(device_name=device_name)


@pytest.mark.smoke
@pytest.mark.parametrize("backend", get_test_backends())
def test_I(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    rgb, aniso = setup_test()
    hsis = HSI()
    hsi = hsis.rgb2hsi3(rgb)
    H, S, I = split_planes(hsi)
    with StopWatcher(f"[generator {N}]", kernel=aniso) as sw:
        with sw.checkpoint("generate"):
            for i, fI in enumerate(aniso.anisodiff(I)):
                if i >= N:
                    break
        hsi = join_planes(H, S, fI)
        rgb2 = hsis.hsi2rgb3(hsi)
        sw.showcheck(rgb2)
    with StopWatcher(f"[{N}]", kernel=aniso) as sw:
        with sw.checkpoint("iterate"):
            fI = aniso.anisodiff_n(I, N)
        hsi = join_planes(H, S, fI)
        rgb2 = hsis.hsi2rgb3(hsi)
        sw.showcheck(rgb2)


@pytest.mark.parametrize("backend", get_test_backends())
def test_IS(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    rgb, aniso = setup_test()
    with StopWatcher(f"[generator {N}", kernel=aniso) as sw:
        with sw.checkpoint("generate"):
            for i, rgb2 in enumerate(aniso.anisodiff_rgbIS(rgb)):
                if i >= N:
                    break
        sw.showcheck(rgb2)
    with StopWatcher(f"[{N}]", kernel=aniso) as sw:
        with sw.checkpoint("iterate"):
            rgb2 = aniso.anisodiff_rgbIS_n(rgb, N)
        sw.showcheck(rgb2)


@pytest.mark.parametrize("backend", get_test_backends())
def test_1plane(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    rgb, aniso = setup_test()
    hsis = HSI()
    hsi = hsis.rgb2hsi3(rgb)
    H, S, I = split_planes(hsi)
    with StopWatcher(f"[{N}]", kernel=aniso) as sw:
        with sw.checkpoint("iterate"):
            fI = aniso.anisodiff_n(I, N)
        hsi = join_planes(H, S, fI)
        rgb2 = hsis.hsi2rgb3(hsi)
        sw.showcheck(rgb2)


@pytest.mark.parametrize("backend", get_test_backends())
def test_2plane(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    if get_backend() == "ocl":
        print("OpenCL does not generally support float2")
        return
    rgb, aniso = setup_test()
    r, g, b = split_planes(rgb)
    rg = join_planes(r, g)
    with StopWatcher(f"[{N}]", kernel=aniso) as sw:
        with sw.checkpoint("iterate"):
            rg2 = aniso.anisodiff_n(rg, N)
        r2, g2 = split_planes(rg2)
        rgb2 = join_planes(r2, g2, b)
        sw.showcheck(rgb2)


@pytest.mark.parametrize("backend", get_test_backends())
def test_3plane(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    device_names = get_available_device_names()
    backend = get_backend()
    if backend == "ocl":
        for device_name in get_available_device_names():
            config = get_device_config(device_name=device_name)
            if config["preferred_vector_width_float"] > 3:
                print(
                    "preferred_vector_width_float",
                    config["preferred_vector_width_float"],
                )
                print(f"The {config['__name__']} OpenCL engine does not support float3")
                continue
            if config.get("compute_capability_major_nv", ""):
                print("Nvidia drivers don't handle float3 well")
                continue
            rgb, aniso = setup_test(device_name=device_name)
            with StopWatcher(
                f"[{N}] on {backend} {device_name[:24]}", kernel=aniso
            ) as sw:
                with sw.checkpoint("iterate"):
                    rgb2 = aniso.anisodiff_n(rgb, N)
                sw.showcheck(rgb2)
    else:
        rgb, aniso = setup_test()
        with StopWatcher(f"[{N}] on {backend}", kernel=aniso) as sw:
            with sw.checkpoint("iterate"):
                rgb2 = aniso.anisodiff_n(rgb, N)
            sw.showcheck(rgb2)


@pytest.mark.parametrize("backend", get_test_backends())
def test_4plane(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    if get_backend() == "ocl":
        from yapocis.ocl.device import get_available_device_names

        for device_name in get_available_device_names():
            rgb, aniso = setup_test(device_name=device_name)
            shape4 = list(rgb.shape)
            shape4[2] = 4
            rgba = np.zeros((shape4), dtype=np.float32)
            rgba[:, :, :3] = rgb
            with StopWatcher(f"[{N}] on {device_name[:24]}", kernel=aniso) as sw:
                with sw.checkpoint("iterate"):
                    rgba2 = aniso.anisodiff_n(rgba, N)
                sw.showcheck(rgba2[:, :, :3])


def _perona_malik_reference(image, n, scale=3.0, step_size=0.2):
    """n Perona-Malik iterations in float64, same scaling and border handling as Anisodiff."""
    img = image.astype(np.float64) * 256.0
    for _ in range(n):
        acc = np.zeros_like(img)
        for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            di = np.roll(img, shift, axis) - img
            cf = di / scale
            acc += di / (1.0 + cf * cf)
        nxt = img.copy()
        nxt[1:-1, 1:-1] = (img + step_size * acc)[1:-1, 1:-1]
        img = nxt
    return img / 256.0


@pytest.mark.parametrize("backend", get_test_backends())
@pytest.mark.parametrize("planes", [1, 3])
@pytest.mark.parametrize("n", [1, 3])
def test_anisodiff_n_runs_n_iterations(backend, planes, n, monkeypatch):
    """anisodiff_n(img, n) used to return the result of n-1 iterations."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    rng = np.random.default_rng(0)
    shape = (37, 53) if planes == 1 else (37, 53, 3)
    image = rng.uniform(0.0, 1.0, shape).astype(np.float32)
    result = Anisodiff().anisodiff_n(image, n)
    expected = _perona_malik_reference(image, n)
    assert result.shape == image.shape
    np.testing.assert_allclose(result, expected, atol=1e-5)


@pytest.mark.parametrize("backend", get_test_backends())
@pytest.mark.parametrize("planes", [1, 3])
def test_anisodiff_generator_matches_reference(backend, planes, monkeypatch):
    """3 planes used to be read with the wrong float3 stride on OpenCL/Metal."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    rng = np.random.default_rng(1)
    shape = (37, 53) if planes == 1 else (37, 53, 3)
    image = rng.uniform(0.001, 0.999, shape).astype(np.float32)
    steps = Anisodiff().anisodiff(image)
    for k in range(1, 4):
        result = next(steps)
        assert result.shape == image.shape
        np.testing.assert_allclose(result, _perona_malik_reference(image, k), atol=1e-5)
