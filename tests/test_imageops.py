#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import numpy as np
from yapocis.lib.imageops import Imageops
from yapocis.ocl.device import get_available_device_names
from yapocis.utils.config import get_device_config
import matplotlib.pyplot as plt

import timeit

def test_clamp():
    imageops = Imageops()
    a = np.random.random(1000).astype(np.float32)
    assert a.min() < .1 and a.max()  > .9
    b = imageops.clamp_pixels(a, .1, .9)
    assert b.min() >= .1 and b.max() <= .9

def test_binarize():
    imageops = Imageops()
    a = np.random.random(1000).astype(np.float32)
    b = imageops.binarize(a, .5)
    assert 450 < b.sum() < 550

def test_stretch():
    imageops = Imageops()
    a = np.random.uniform(.1, .9, (2000,2000), ).astype(np.float32)
    assert a.min() >= .1 and a.max() <= .6
    b = imageops.stretch(a, a.min(), a.max(), 0., 1.0)
    assert b.min() == 0.0 and b.max() == 1.0

def test_histogram():
    NBINS=1000
    imageops = Imageops()
    a = np.random.uniform(.1, .9, (2000,2000), ).astype(np.float32)
    h = imageops.histogram(a)
    assert h.sum() == a.size
    h = imageops.histogram(a, n_bins=NBINS)
    assert h.sum() == a.size

def test_ops():
    NBINS=1000
    imageops = Imageops()
    a = np.random.normal(.5, .1, (2000,2000), ).astype(np.float32)
    b = imageops.clamp_pixels(a, 0, 1-1e-05)
    h = imageops.histogram(a, n_bins=NBINS)
    x = np.arange(0,1,1/NBINS)
    plt.plot(x,h,label="full")
    a = imageops.stretch(a,.15,.85, 0, 1)
    h = imageops.histogram(a, n_bins=NBINS)
    plt.plot(x,h,label="stretch")
    plt.legend()
    plt.show()


def test_histogram_performance():
    ITERATIONS=10
    nbins=256
    for device_name in get_available_device_names():
        cfg = get_device_config(device_name)
        compute_units = cfg["max_compute_units"]
        print()
        print(device_name)
        imageops = Imageops(device_name=device_name)
        histogram_res = imageops.histogram_res
        a = np.random.uniform(.1, .9, (2000, 2000)).astype(np.float32)
        a *= .5
        a += .1
        n_workers = [compute_units, compute_units*4, compute_units*16, compute_units*32]
        for nw in reversed(n_workers):
            partials = np.zeros((nw, 256), dtype=np.int32)
            imageops.write(a)
            imageops.write(partials)
            print(f"{nw:03d}", min(*timeit.repeat("histogram_res(a,partials)",number=ITERATIONS, globals=locals(),repeat=3)))


