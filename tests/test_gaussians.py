#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.
import numpy as np
import time

from yapocis.rpc.kernel_manager import load_program
from yapocis.utils.basic import show_array
from yapocis.rpc.device import get_available_device_names

from yapocis.lib.gaussian import Gaussian, GaussianWeights, are_close


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
        show_array(f"gauss scale {scale}", b)


def test_zcs():
    g = Gaussian()
    image = np.zeros((512, 512), dtype=np.float32)
    image[255, 255] = 1.0
    smaller = g.gauss_image(image, 2)
    larger = g.gauss_image(image, 3)
    dog = smaller - larger
    zca = g.zcs.zcs1(dog)
    show_array("gauss 2", smaller)
    show_array("gauss 3", larger)
    show_array("dog", dog)
    show_array("zca", zca)

def test_zcs_scales():
    g = Gaussian()
    for res in (False, True):
        subtotal = 0.0
        for scale in range(g.get_scales() - 1):
            a = np.zeros((400, 400), dtype=np.float32)
            a[200, :] = 1.0
            a[:, 200] = 1.0
            t = time.time()
            zca, grad, theta = g.zcsdog(a, scale)
            subtotal += (time.time() - t)
            # show_array("org %s" % scale, a)
            # show_array("zcs %s" % scale, zca)
            # show_array("grad %s" % scale, grad)
            show_array("theta %s" % scale, theta)
        print("Res", res, "seconds", subtotal)


def test_gradient():
    a = np.random.sample((1000, 1000)).astype(np.float32)
    for engine_name in get_available_device_names():
        if isinstance(engine_name,int):
            continue
        program = load_program("gradient", device_name=engine_name)
        t = time.time()
        gr,th = program.gradient(a, 3)
        elapsed = time.time()-t
        show_array(f"Gradient {engine_name} {elapsed}", gr)
