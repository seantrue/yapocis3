#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import numpy as np
import time

import yapocis.rpc.programs
from yapocis.utils.basic import show_array, stage, imread
from yapocis.lib.hsi import rgb2hsi, hsi2rgb
from yapocis.lib.channels import split_channels, join_channels, split_planes, join_planes

from yapocis.rpc.engines import get_available

from yapocis.lib.zcs import zcs
from yapocis.lib.gaussian import gauss_1d, are_close, gaussians, get_scales, get_gaussian_kernel, \
    get_gaussian, gauss_image, zcsdog
from yapocis.lib.operators import add, add_res, program, sub, sub_res, mul, mul_res
from yapocis.lib.median import median3x3fast, median3x3slow
from yapocis.lib.anisodiff import anisodiff, anisodiff_rgbIS
from yapocis.utils.config import get_device_config, dump_config

from . import test_image_path

test_image = test_image_path()


def test_gaussian_1d():
    "Make sure that the kernel sums back to 1"
    g = gauss_1d()
    assert are_close(g.sum(), 1.0)


def test_gaussians():
    mw = 100
    gs = gaussians(maxwidth=mw)
    for g in gs:
        assert len(g) < mw


def test_kernels():
    for scale in range(get_scales()):
        gkernel = get_gaussian_kernel(scale)
        a = np.zeros((256, 256), dtype=np.float32)
        a[127, 127] = 1.0
        b = gkernel(1, a)
        gsum = get_gaussian(scale).sum()
        assert are_close(b[127, :].sum(), gsum)
        b = gkernel(0, a)
        assert are_close(b[:, 127].sum(), gsum)
        # Compare in order of lowest precision
        assert are_close(a.sum(), 1.0)


def test_gauss_image():
    for scale in range(get_scales()):
        a = np.zeros((400, 700), dtype=np.float32)
        a[200, :] = 1.0
        a[:, 200] = 1.0
        for xy in range(400):
            a[xy, xy] = 1.0
        b = gauss_image(a, scale)
        show_array(f"gauss scale {scale}", b)


def test_zcs():
    for res in (False, True):
        subtotal = 0.0
        for scale in range(get_scales() - 1):
            a = np.zeros((1200, 1800), dtype=np.float32)
            a[200, :] = 1.0
            a[:, 200] = 1.0
            t = time.time()
            zca, grad, theta = zcsdog(a, scale)
            subtotal += (time.time() - t)
            show_array("org %s" % scale, a)
            show_array("zcs %s" % scale, zca)
            show_array("grad %s" % scale, grad)
            show_array("theta %s" % scale, theta)
        print("Res", res, "seconds", subtotal)


def test_gradient():
    a = np.random.sample((1000, 1000)).astype(np.float32)
    t = time.time()
    b = np.gradient(a)
    print("Numpy seconds", time.time() - t)
    for engine_name in get_available().keys():
        program = yapocis.rpc.programs.load_program("gradient", engine_name=engine_name)
        t = time.time()
        c = program.gradient(a, 1)
        print("Engine %s seconds %s" % (engine_name, time.time() - t))


def test_hsi():
    r = np.empty((256, 256), dtype=np.float32)
    g = np.empty_like(r)
    b = np.empty_like(r)
    for i in range(256):
        r[i, :] = i
        g[:, i] = i
        b[i, :] = 255 - i
    rgb = join_channels(r, g, b)
    rgb /= 255.0
    show_array("Test data", rgb)
    r, g, b = split_channels(rgb)
    rgb2 = join_channels(r, g, b)
    show_array("Test split and join", rgb2)
    h, s, i = rgb2hsi(r, g, b)
    show_array("I", i)
    show_array("H", h)
    show_array("S", s)
    r, g, b = hsi2rgb(h, s, i)
    rgb3 = join_channels(r, g, b)
    show_array("from HSI", rgb3)
    print(program)


def test_median3():
    a1 = np.random.sample((500, 500)).astype(np.float32)
    a2 = a1.copy()
    stage("slow")
    b1 = median3x3slow(a1, 500)
    stage()
    stage("fast")
    b2 = median3x3fast(a2, 500)
    stage()
    error = np.sum(np.abs(b1.flatten() - b2.flatten()))
    assert error == 0.0


def test_operators():
    # 1+0 -> 1
    a = np.ones((100, 100), dtype=np.float32)
    b = np.zeros_like(a)
    c = add(a, b)
    d = np.empty_like(a)
    add_res(a, b, d)
    assert c.sum() == c.size * 1.0
    program.read(d)
    assert a.sum() == c.sum()
    assert a.sum() == d.sum()
    # 1-1 == 0
    b[:, :] = 1.0
    c = sub(a, b)
    d[:, :] = 10
    sub_res(a, b, d)
    program.read(d)
    assert c.sum() == 0.0
    assert c.sum() == d.sum()
    # 1*1 = 1
    c = mul(a, b)
    d[:, :] = 10
    mul_res(a, b, d)
    program.read(d)
    assert a.sum() == c.sum()
    assert a.sum() == d.sum()
    # 1/1 = 1
    # c = div(a,b)
    # d[:,:]=10
    # div_res(a,b,d)
    # program.read(d)
    # assert a.sum() == c.sum()
    # assert a.sum() == d.sum()


def test_zcs():
    image = np.zeros((512, 512), dtype=np.float32)
    image[255, 255] = 1.0
    smaller = gauss_image(image, 2)
    larger = gauss_image(image, 3)
    dog = smaller - larger
    zca = zcs(dog)
    show_array("gauss 2", smaller)
    show_array("gauss 3", larger)
    show_array("dog", dog)
    show_array("zca", zca)


def test_perona_malik():
    rgb = imread(test_image).astype(np.float32) / 256
    show_array("org IS", rgb)
    for i, rgb2 in enumerate(anisodiff_rgbIS(rgb)):
        if i >= 50:
            break
    show_array(f"filtered IS", rgb2)

def test_perona_malik():
    rgb = imread(test_image).astype(np.float32) / 256
    show_array("org IS", rgb)
    for i, rgb2 in enumerate(anisodiff_rgbIS(rgb)):
        if i >= 50:
            break
    show_array(f"filtered IS", rgb2)

def test_perona2_malik():
    config = get_device_config()
    if config["native_vector_width_float"] > 2:
        print(config["preferred_vector_width_float"])
        raise RuntimeError("This OpenCL engine does not support float2")
    rgb = imread(test_image).astype(np.float32) / 256
    show_array("org rg", rgb)
    r,g,b = split_planes(rgb)
    rg = join_planes(r,g)
    for i, rgb2 in enumerate(anisodiff(rg)):
        if i >= 50:
            break
    r,g = split_planes(rg)
    rgb2 = join_planes(r,g,b)
    show_array(f"filtered rg", rgb2)


def test_perona3_malik():
    config = get_device_config()
    if config["preferred_vector_width_float"] > 3:
        print(config["preferred_vector_width_float"])
        raise RuntimeError("This OpenCL engine does not support float3")
    rgb = imread(test_image).astype(np.float32) / 256
    show_array("org rgb", rgb)
    for i, rgb2 in enumerate(anisodiff(rgb)):
        if i >= 50:
            break
    show_array(f"filtered rgb", rgb2)


def test_perona4_malik():
    rgb = imread(test_image).astype(np.float32) / 256
    show_array("org rgba", rgb)
    shape4 = list(rgb.shape)
    shape4[2] = 4
    rgba = np.zeros((shape4), dtype=np.float32)
    rgba[:, :, :3] = rgb
    for i, rgba2 in enumerate(anisodiff(rgba)):
        if i >= 50:
            break
    show_array(f"filtered rgba", rgba2[:, :, :3])
