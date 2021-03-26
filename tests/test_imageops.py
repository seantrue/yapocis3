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
    assert a.min() >= .1 and a.max() <= .9
    b = imageops.stretch(a, a.min(), a.max(), 0., 1.0)
    assert b.min() <=1e-06 and b.max() >= 1.0-1e-06

def test_histogram():
    imageops = Imageops(FASTBINS=0)
    a = np.random.uniform(.1, .9, (2000,2000), ).astype(np.float32)
    h = imageops.histogram(a, n_bins=256)
    h1 = imageops.histogram(a, n_bins=1500)
    assert h.sum() == h1.sum()

def test_ops():
    NBINS=1000
    imageops = Imageops(FASTBINS=0)
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
        if "GeForce" in device_name:
            FASTBINS=1024
            local_size = (1,)
        else:
            FASTBINS=False
            local_size = None
        compute_units = cfg["max_compute_units"]
        print()
        print(device_name, compute_units)
        FBS = [FASTBINS]
        if FASTBINS:
            FBS.insert(0,0)
        for FB in FBS:
            imageops = Imageops(device_name=device_name,FASTBINS=FB)
            histogram_res = imageops.histogram_res
            a = np.random.uniform(.1, .9, (2000, 2000)).astype(np.float32)
            n_workers = [compute_units, compute_units*4, compute_units*16, compute_units*32]
            if FASTBINS:
                n_workers += [compute_units*64, compute_units*96, compute_units*128,compute_units*192]
            for nw in reversed(n_workers):
                partials = np.zeros((nw, 256), dtype=np.int32)
                imageops.write(a)
                imageops.write(partials)
                statement = f"histogram_res(a,partials)"
                print(f"{nw:03d} {FB}", min(*timeit.repeat(statement,number=ITERATIONS, globals=locals(),repeat=3)))
            print()


