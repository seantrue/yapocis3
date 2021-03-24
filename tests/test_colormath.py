#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from yapocis.lib.channels import split_channels, join_channels
from yapocis.utils.stereo_typing import *

from yapocis.lib.hsi import HSI
from yapocis.lib.colormath import Colormath, transform_names
from yapocis.utils.basic import histeq

from . import ASIZE, show_snapshots, snapshow, show, image_test, StopWatcher


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
    show("RGB in", rgb)
    r, g, b = split_channels(rgb)
    rgb2 = join_channels(r, g, b)
    with StopWatcher() as shower:
        h, s, i = hsis.rgb2hsi(r, g, b)
        hsi3 = hsis.rgb2hsi3(rgb)
        h3,s3,i3 = split_channels(hsi3)
        rgb3 = hsis.hsi2rgb3(hsi3)
        rgb4 = np.zeros((ASIZE,ASIZE,4), dtype=np.float32)
        rgb4[:,:,:3] = rgb[:,:,:]
        hsi4 = hsis.rgb2hsi3(rgb4)
        rgb5 = hsis.hsi2rgb3(hsi4)
        rgb5 = rgb5[:,:,:3]
        shower(hsis, "RGB out", rgb5)
    assert np.sum(rgb-rgb2) == 0.0
    assert abs(np.sum(rgb.ravel()-rgb3.ravel()))/rgb.size < .00001
    assert abs(np.sum(rgb.ravel()-rgb5.ravel()))/rgb.size < .00001
    assert np.sum(i-i3) == 0.0
    assert np.sum(s-s3) == 0.0
    assert np.sum(h-h3) == 0.0


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
    with StopWatcher() as shower:
        xyz = cm.transform(rgb, 0, global_size=rgb.size // 3)
        rgb2 = cm.transform_named("adobe1998_d65.xyz2rgb", xyz)
        shower("RGB out", rgb2)

def test_xyz_view():
    cm = Colormath()
    rgb = image_test()
    xyz = cm.transform_named("srgb_d50.rgb2xyz", rgb)
    show("xyz", xyz)
    x, y, z = split_channels(xyz)
    ones = np.ones_like(x)*.3
    xyz = join_channels(x, y, z)
    show("RGB in", rgb)
    with StopWatcher() as shower:
        rgb2 = cm.transform_named("srgb_d50.xyz2rgb", xyz)
        xyz = join_channels(ones,y, z)
        rgb3 = cm.transform_named("srgb_d50.xyz2rgb", xyz)
        xyz = join_channels(x, ones, z)
        rgb4 = cm.transform_named("srgb_d50.xyz2rgb", xyz)
        xyz = join_channels(x, y, ones)
        rgb5 = cm.transform_named("srgb_d50.xyz2rgb", xyz)
        shower("RGB out", rgb2)
    show("x=1", rgb3)
    show("y=1", rgb4)
    show("z=1", rgb5)
    show("Y", y)


def test_colorspaces():
    cm = Colormath()
    rgb = image_test()
    show("RGB in", rgb)
    transformed = []
    with StopWatcher() as shower:
        xyz = cm.transform_named("srgb_d50.rgb2xyz", rgb)
        for name in transform_names[1::2]:
            rgb2 = cm.transform_named(name, xyz)
            transformed.append((name, rgb2))
        shower(cm, *transformed[0])
    for name, rgb2 in transformed[1:]:
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
    with StopWatcher() as shower:
        xyz = cm.transform_named("srgb_d50.rgb2xyz", rgb)
        xyY = cm.xyz2xyY(xyz)
        xyz2 = cm.xyY2xyz(xyY)
        rgb2 = cm.transform_named("adobe1998_d65.xyz2rgb", xyz2)
        show("RGB out", rgb2)
    show("xyY", xyY)
