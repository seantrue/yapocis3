#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.
import numpy as np
import time

from yapocis.rpc.device import get_available_device_names

from yapocis.lib.gaussian import Gaussian, GaussianWeights, are_close
from yapocis.lib.gradient import Gradient
from yapocis.lib.hsi import HSI
from yapocis.lib.channels import split_planes
from . import show_snapshots, snapshow, show, image_test

def test_gaussian_1d():
    "Make sure that the kernel sums back to 1"
    g = GaussianWeights.compute_gaussian_weight()
    assert are_close(g.sum(), 1.0)


def test_gaussians():
    mw = 100
    gws = GaussianWeights(max_width=mw)
    for g in gws.gws:
        assert len(g) < mw


def test_gaussian_kernels():
    g = Gaussian()
    for scale in range(g.get_scales()):
        gkernel = g.get_gaussian_kernel(scale)
        a = np.zeros((256, 256), dtype=np.float32)
        a[127, 127] = 1.0
        b = gkernel(1, a)
        gsum = g.gaussian_weights.get_gaussian_weight(scale).sum()
        assert are_close(b[127, :].sum(), gsum)
        b = gkernel(0, a)
        assert are_close(b[:, 127].sum(), gsum)
        # Compare in order of lowest precision
        assert are_close(a.sum(), 1.0)


def test_gauss_image():
    g = Gaussian()
    for scale in range(g.get_scales()):
        a = np.zeros((400, 700), dtype=np.float32)
        a[200, :] = 1.0
        a[:, 200] = 1.0
        for xy in range(400):
            a[xy, xy] = 1.0
        b = g.gauss_image(a, scale)
        snapshow(g, f"scale {scale}", b)


def test_zcs():
    g = Gaussian()
    image = np.zeros((512, 512), dtype=np.float32)
    image[255, 255] = 1.0
    smaller = g.gauss_image(image, 4)
    larger = g.gauss_image(image, 5)
    dog = smaller - larger
    zca = g.zcs(dog)
    show("scale 4", smaller)
    show("scale 5", larger)
    show("dog", dog)
    snapshow(g, zca)

def test_zcs_scales():
    for device_name in get_available_device_names():
        g = Gaussian(device_name=device_name)
        subtotal = 0.0
        for scale in range(g.get_scales() - 1):
            a = np.zeros((400, 400), dtype=np.float32)
            a[200, :] = 1.0
            a[:, 200] = 1.0
            t = time.time()
            zca, grad, theta = g.zcsdog(a, scale, res=True)
            subtotal += (time.time() - t)
            snapshow(g, f"{subtotal:.03f}", zca)
        print(f" {device_name} {subtotal:.03f} seconds")


def test_gradient():
    img = image_test()
    for device_name in get_available_device_names():
        hsi = HSI(device_name=device_name)
        g = Gradient(device_name=device_name)
        _,_,I = hsi.rgb2hsi(*split_planes(img))
        gr,th = g.gradient(I, 3)
        show(device_name, "theta",th)
        snapshow(g, device_name, "grad", gr)
