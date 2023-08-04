#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

import numpy as np

from yapocis.lib.imageops import Imageops


def test_clip():
    imageops = Imageops()
    a = np.random.random(1000).astype(np.float32)
    assert a.min() < .1 and a.max()  > .9
    b = imageops.clip(a, .1, .9)
    assert b.min() >= .1 and b.max() <= .9

def test_binarize():
    imageops = Imageops()
    a = np.random.random(1000).astype(np.float32)
    b = imageops.binarize(a, .5)
    assert 450 < b.sum() < 550

def test_stretch():
    imageops = Imageops()
    a = np.random.random(1000).astype(np.float32)
    a *= .5
    a += .1
    assert a.min() >= .1 and a.max() <= .6
    b = imageops.stretch(a, a.min(), a.max(), 0., 1.0)
    assert b.min() == 0.0 and b.max() >= 0.999999
