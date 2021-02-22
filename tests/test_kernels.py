#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import numpy as np

from yapocis.utils.basic import show_array, stage
from yapocis.lib.hsi import rgb2hsi, rgb2hsi3, hsi2rgb3
from yapocis.lib.channels import split_channels, join_channels

from yapocis.lib.operators import add, add_res, program, sub, sub_res, mul, mul_res
from yapocis.lib.median import Median

from . import image_path_test

def test_hsi():
    r = np.empty((256, 256), dtype=np.float32)
    g = np.empty_like(r)
    b = np.empty_like(r)
    for i in range(256):
        r[i, :] = i
        g[:, i] = i
        b[i, :] = 255 - i
    rgb = join_channels(r, g, b)
    rgb /= 256.0
    show_array("Test data", rgb)
    r, g, b = split_channels(rgb)
    rgb2 = join_channels(r, g, b)
    assert np.sum(rgb-rgb2) == 0.0
    h, s, i = rgb2hsi(r, g, b)
    hsi3 = rgb2hsi3(rgb)
    h3,s3,i3 = split_channels(hsi3)
    assert np.sum(i-i3) == 0.0
    assert np.sum(s-s3) == 0.0
    assert np.sum(h-h3) == 0.0
    rgb3 = hsi2rgb3(hsi3)
    assert abs(np.sum(rgb.ravel()-rgb3.ravel()))/rgb.size < .00001
    rgb4 = np.zeros((256,256,4), dtype=np.float32)
    rgb4[:,:,:3] = rgb[:,:,:]
    hsi4 = rgb2hsi3(rgb4)
    rgb5 = hsi2rgb3(hsi4)
    rgb5 = rgb5[:,:,:3]
    assert abs(np.sum(rgb.ravel()-rgb5.ravel()))/rgb.size < .00001
    show_array("from HSI4", rgb5)
    print(program)


def test_median3():
    median = Median()
    a1 = np.random.sample((500, 500)).astype(np.float32)
    a2 = a1.copy()
    stage("slow")
    b1 = median.median3x3slow(a1, 500)
    stage()
    stage("fast")
    b2 = median.median3x3fast(a2, 500)
    stage()
    error = np.sum(np.abs(b1.ravel() - b2.ravel()))
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



