#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

from yapocis.lib.channels import join_channels, split_channels
from yapocis.lib.colormath import Colormath, transform_names
from yapocis.lib.hsi import HSI
from yapocis.utils.stereo_typing import *

from . import ASIZE, StopWatcher, image_test, show


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
    with StopWatcher("hsi") as sw:

        def checkshow(a, checkpoint):
            sw.checkpoint(checkpoint)
            sw.show(hsis, a, checkpoint, depth=1)

        h, s, i = hsis.rgb2hsi(r, g, b)
        sw.checkpoint("rgb2hsi")
        hsi3 = hsis.rgb2hsi3(rgb)
        checkshow(hsi3, "rgb2hsi3")
        rgb3 = hsis.hsi2rgb3(hsi3)
        checkshow(rgb3, "hsi2rgb3")
        rgb4 = np.zeros((ASIZE, ASIZE, 4), dtype=np.float32)
        rgb4[:, :, :3] = rgb[:, :, :]
        hsi4 = hsis.rgb2hsi3(rgb4)
        checkshow(hsi4, "rgb2hsi3")
        rgb5 = hsis.hsi2rgb3(hsi4)
        rgb5 = rgb5[:, :, :3]
        checkshow(rgb5, "hsi2rgb3")
        print(sw.summary())

    assert np.sum(rgb - rgb2) == 0.0
    assert abs(np.sum(rgb.ravel() - rgb3.ravel())) / rgb.size < 0.00001
    assert abs(np.sum(rgb.ravel() - rgb5.ravel())) / rgb.size < 0.00001
    h3, s3, i3 = split_channels(hsi3)
    assert np.sum(i - i3) == 0.0
    assert np.sum(s - s3) == 0.0
    assert np.sum(h - h3) == 0.0


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
    with StopWatcher("xyz") as sw:
        xyz = cm.transform(rgb, 0, global_size=rgb.size // 3)
        rgb2 = cm.transform_named("adobe1998_d65.xyz2rgb", xyz)
        sw.checkpoint("roundtrip")
        sw.show(cm, rgb2, "roundtrip")
        print(sw.summary())


def test_xyz_view():
    cm = Colormath()
    rgb = image_test()
    with StopWatcher("srgb_d50") as sw:
        xyz = cm.transform_named("srgb_d50.rgb2xyz", rgb)
        sw.checkpoint("xzy")
        sw.show(cm, xyz, "2xyz")
        x, y, z = split_channels(xyz)
        sw.show(cm, x, "X")
        sw.show(cm, y, "Y")
        sw.show(cm, z, "Z")
        ones = np.ones_like(x) * 0.3

        def checkshow(x1, y1, z1, checkpoint):
            a = join_channels(x1, y1, z1)
            rgbout = cm.transform_named("srgb_d50.xyz2rgb", xyz)
            sw.checkpoint(checkpoint)
            sw.show(cm, a, checkpoint, depth=1)

        checkshow(x, y, z, "2rgb")
        checkshow(ones, y, z, "1yz")
        checkshow(x, ones, z, "x1z")
        checkshow(x, y, ones, "xy1")
        print(sw.summary())


def test_colorspaces():
    cm = Colormath()
    rgb = image_test()
    transformed = []
    with StopWatcher("colormath") as sw:
        sw.show(cm, rgb, "rgb")
        sw.checkpoint("show")

        def checkshow(a, checkpoint):
            sw.checkpoint(checkpoint)
            sw.show(cm, a, checkpoint, depth=1)

        xyz = cm.transform_named("srgb_d50.rgb2xyz", rgb)
        checkshow(xyz, "xyz")
        for name in transform_names[1::2]:
            rgb2 = cm.transform_named(name, xyz)
            checkshow(rgb2, name)
        print(sw.summary())


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
    with StopWatcher("xyY") as sw:
        xyz = cm.transform_named("srgb_d50.rgb2xyz", rgb)
        sw.checkpoint("xyz")
        xyY = cm.xyz2xyY(xyz)
        sw.checkpoint("xyY")
        xyz2 = cm.xyY2xyz(xyY)
        sw.checkpoint("xyY")
        rgb2 = cm.transform_named("adobe1998_d65.xyz2rgb", xyz2)
        sw.checkpoint("2rgb")
        sw.show(cm, xyY, "xyY")
        sw.show(cm, rgb2, "RGB out")
        print(sw.summary())
