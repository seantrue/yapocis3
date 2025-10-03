#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

import numpy as np

from yapocis.lib.channels import split_planes
from yapocis.lib.gaussian import Gaussian, GaussianWeights, are_close
from yapocis.lib.gradient import Gradient
from yapocis.lib.hsi import HSI
from yapocis.ocl.device import get_available_device_names
from . import StopWatcher, image_test


def test_weight_sum():
    "Make sure that the kernel sums back to 1"
    g = GaussianWeights.compute_gaussian_weight()
    assert are_close(g.sum(), 1.0)


def test_weights():
    mw = 100
    gws = GaussianWeights(max_width=mw)
    for g in gws.gws:
        assert len(g) < mw


def test_kernels():
    g = Gaussian()
    a = np.zeros((400, 401), dtype=np.float32)
    a[127, 127] = 1.0
    with StopWatcher("impulse ") as sw:
        for scale in range(g.get_scales()):
            checkpoint = f"{scale}"
            b = g.gauss_image(a, scale)
            sw.checkpoint(checkpoint)
            sw.show(g, b, checkpoint)
            gsum = g.gaussian_weights.get_gaussian_weight(scale).sum()
            bsum = b.sum()
            assert are_close(gsum, bsum)
            assert are_close(bsum, 1.0)
            sw.checkpoint("check")


def test_image():
    g = Gaussian()
    a = np.zeros((400, 401), dtype=np.float32)
    a[200, :] = 1.0
    a[:, 200] = 1.0
    for xy in range(400):
        a[xy, xy] = 1.0
    with StopWatcher("gauss_image") as sw:
        for scale in range(g.get_scales()):
            checkpoint = f"{scale=}"
            b = g.gauss_image(a, scale)
            sw.checkpoint(checkpoint)
            sw.show(g, b, checkpoint)
        print(sw.summary())


def test_zcs():
    g = Gaussian()
    image = np.zeros((512, 513), dtype=np.float32)
    image[255, 240:280] = 1.0

    with StopWatcher("zcs") as sw:
        for scale in range(g.get_scales() - 1):
            checkpoint = f"{scale=}"
            smaller = g.gauss_image(image, scale)
            larger = g.gauss_image(image, scale + 1)
            dog = smaller - larger
            zca = g.zcs(dog)
            sw.checkpoint(checkpoint)
            sw.show(g, zca, checkpoint)
            sw.show(g, dog, "dog", checkpoint)
        print(sw.summary())


def test_multiscale_zcs():
    a = np.zeros((400, 401), dtype=np.float32)
    a[150:250, 150:250] = 1.0
    for device_name in get_available_device_names():
        g = Gaussian(device_name=device_name)
        with StopWatcher(device_name) as sw:
            for scale in range(g.get_scales() - 1):
                checkpoint = f"{scale=}"
                zca, grad, theta = g.zcsdog(a, scale, res=True)
                sw.checkpoint(checkpoint)
                sw.show(g, zca, f"{scale=}")
            print(sw.summary())


# def test_thresholds():
#     g = Gaussian()
#     subtotal = 0.0
#     for threshold in np.linspace(7.4e-06,7.6e-06,5):
#         for scale in range(g.get_scales() - 1):
#             a = np.zeros((400, 401), dtype=np.float32)
#             a[200, :] = 1.0
#             a[:, 200] = 1.0
#             zca, grad, theta = g.zcsdog(a, scale, threshold=threshold)
#             snapshow(g,  f"{scale} {threshold}", zca)


def test_gradient():
    img = image_test()
    for device_name in get_available_device_names():
        hsi = HSI(device_name=device_name)
        g = Gradient(device_name=device_name)
        _, _, I = hsi.rgb2hsi(*split_planes(img))
        with StopWatcher() as sw:
            gr, th = g.gradient(I, 3)
            sw.checkpoint("gradient")
            sw.show(g, th, "theta", th)
            sw.show(g, gr, device_name)
            print(sw.summary())


def test_multiscale_zcs_test():
    hsis = HSI()
    img = image_test()
    h, s, i = split_planes(hsis.hsi2rgb3(img))
    g = Gaussian(device_name=0)
    subtotal = 0.0
    with StopWatcher() as sw:
        for scale in range(g.get_scales() - 1):
            checkpoint = f"{scale=}"
            zca, grad, theta = g.zcsdog(i, scale, res=True)
            sw.show(g, zca, checkpoint)
        sw.show(g, grad, "grad")
        sw.show(g, theta, "theta")
        print(sw.summary())
