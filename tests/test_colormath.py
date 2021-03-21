#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from yapocis.lib.channels import split_channels, join_channels
from yapocis.utils.typing import *

from yapocis.lib.hsi import HSI
from yapocis.lib.colormath import Colormath, transform_names
from yapocis.utils.basic import histeq

from . import ASIZE, show_snapshots, snapshow, show, image_test


def test_hsi():
    hsis = HSI()
    r = np.empty((ASIZE, ASIZE), dtype=np.float32)
    g = np.empty_like(r)
    b = np.empty_like(r)
    for i in range(ASIZE):
        r[i, :] = i
        g[:, i] = i
        b[i, :] = ASIZE - i
    rgb = join_channels(r, g, b)
    rgb /= ASIZE
    show("Test data", rgb)
    r, g, b = split_channels(rgb)
    rgb2 = join_channels(r, g, b)
    assert np.sum(rgb - rgb2) == 0.0
    h, s, i = hsis.rgb2hsi(r, g, b)
    hsi3 = hsis.rgb2hsi3(rgb)
    h3, s3, i3 = split_channels(hsi3)
    assert np.sum(i - i3) == 0.0
    assert np.sum(s - s3) == 0.0
    assert np.sum(h - h3) == 0.0
    rgb3 = hsis.hsi2rgb3(hsi3)
    assert abs(np.sum(rgb.ravel() - rgb3.ravel())) / rgb.size < .00001
    rgb4 = np.zeros((ASIZE, ASIZE, 4), dtype=np.float32)
    rgb4[:, :, :3] = rgb[:, :, :]
    hsi4 = hsis.rgb2hsi3(rgb4)
    rgb5 = hsis.hsi2rgb3(hsi4)
    rgb5 = rgb5[:, :, :3]
    assert abs(np.sum(rgb.ravel() - rgb5.ravel())) / rgb.size < .00001
    snapshow(hsis, rgb5)


def test_xyz():
    cm = Colormath()
    r = np.empty((ASIZE, ASIZE), dtype=np.float32)
    g = np.empty_like(r)
    b = np.empty_like(r)
    for i in range(ASIZE):
        r[i, :] = i
        g[:, i] = i
        b[i, :] = ASIZE - i
    rgb = join_channels(r, g, b)
    rgb /= ASIZE
    show("RGB in", rgb)
    rgb = np.ascontiguousarray(rgb)
    xyz = cm.transform(rgb, 0, global_size=rgb.size // 3)
    rgb2 = cm.transform_named("adobe1998_d65.xyz2rgb", xyz)
    show("RGB out", rgb)
    snapshow(cm, rgb2)


def test_xyz_view():
    cm = Colormath()
    rgb = image_test()
    xyz = cm.transform_named("srgb_d50.rgb2xyz", rgb)
    show("xyz", xyz)
    x, y, z = split_channels(xyz)
    ones = np.ones_like(x)*.3
    xyz = join_channels(x, y, z)
    show("RGB in", rgb)
    rgb2 = cm.transform_named("srgb_d50.xyz2rgb", xyz)
    show("RGB out", rgb2)
    xyz = join_channels(ones,y, z)
    rgb2 = cm.transform_named("srgb_d50.xyz2rgb", xyz)
    show("x=1", rgb2)
    xyz = join_channels(x, ones, z)
    rgb2 = cm.transform_named("srgb_d50.xyz2rgb", xyz)
    show("y=1", rgb2)
    xyz = join_channels(x, y, ones)
    rgb2 = cm.transform_named("srgb_d50.xyz2rgb", xyz)
    show("z=1", rgb2)
    snapshow(cm, y)


def test_colorspaces():
    cm = Colormath()
    rgb = image_test()
    show("RGB in", rgb)
    xyz = cm.transform_named("srgb_d50.rgb2xyz", rgb)
    for name in transform_names[1::2]:
        rgb2 = cm.transform_named(name, xyz)
        show(name, rgb2)

def test_xyY():
    cm = Colormath()
    r = np.empty((ASIZE, ASIZE), dtype=np.float32)
    g = np.empty_like(r)
    b = np.empty_like(r)
    for i in range(ASIZE):
        r[i, :] = i
        g[:, i] = i
        b[i, :] = ASIZE - i
    rgb = join_channels(r, g, b)
    rgb /= ASIZE
    show("RGB in", rgb)
    rgb = np.ascontiguousarray(rgb)
    xyz = cm.transform_named("srgb_d50.rgb2xyz", rgb)
    xyY = cm.xyz2xyY(xyz)
    xyz2 = cm.xyY2xyz(xyY)
    rgb2 = cm.transform_named("adobe1998_d65.xyz2rgb", xyz2)
    show("RGB out", rgb2)
    snapshow(cm, xyY)
