#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.

import numpy as np

from yapocis.lib.channels import join_channels, split_channels
from yapocis.lib.hsi import HSI
from yapocis.lib.median import Median
from yapocis.lib.operators import Operators
from yapocis.utils.basic import show_array, stage
from yapocis.utils.stereo_typing import *

from . import (ASIZE, StopWatcher, image_path_test, show, show_snapshots,
               snapshot, snapshow)


def test_median3():
    median = Median()
    a1 = np.random.sample((ASIZE, ASIZE)).astype(np.float32)
    a2 = a1.copy()
    stage("slow")
    b1 = median.median3x3slow(a1, 500)
    stage()
    stage("fast")
    b2 = median.median3x3fast(a2, 500)
    stage()
    error = np.sum(np.abs(b1.ravel() - b2.ravel()))
    assert error == 0.0
    snapshot(median)


def test_operators():
    # 1+0 -> 1
    ops = Operators()
    a = np.ones((ASIZE, ASIZE), dtype=np.float32)
    b = np.zeros_like(a)
    c = ops.add(a, b)
    d = np.empty_like(a)
    ops.add_res(a, b, d)
    assert c.sum() == c.size * 1.0
    ops.read(d)
    assert a.sum() == c.sum()
    assert a.sum() == d.sum()
    # 1-1 == 0
    b[:, :] = 1.0
    c = ops.sub(a, b)
    d[:, :] = 10
    ops.sub_res(a, b, d)
    ops.read(d)
    assert c.sum() == 0.0
    assert c.sum() == d.sum()
    # 1*1 = 1
    c = ops.mul(a, b)
    d[:, :] = 10
    ops.mul_res(a, b, d)
    ops.read(d)
    assert a.sum() == c.sum()
    assert a.sum() == d.sum()
    snapshot(ops)



