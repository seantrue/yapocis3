#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.

from yapocis.lib.channels import join_channels, split_channels
from yapocis.lib.colormath import Colormath, transform_names
from yapocis.lib.hsi import HSI
from yapocis.utils.stereo_typing import *
from . import ASIZE, StopWatcher, image_test, show, assert_closeenough


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
    r, g, b = split_channels(rgb)
    rgb2 = join_channels(r, g, b)
    with StopWatcher("hsi", kernel=hsis) as sw:
        sw.show(rgb, "RGB in")
        with sw.checkpoint("rgb2hsi"):
            h, s, i = hsis.rgb2hsi(r, g, b)
        sw.showcheck(h, "h")
        sw.showcheck(s, "s")
        sw.showcheck(i, "i")
        with sw.checkpoint("rgb2hsi3"):
            hsi3 = hsis.rgb2hsi3(rgb)
        sw.showcheck(hsi3)
        with sw.checkpoint("hsi2rgb3"):
            rgb3 = hsis.hsi2rgb3(hsi3)
        sw.showcheck(rgb3)
        rgb4 = np.zeros((ASIZE, ASIZE, 4), dtype=np.float32)
        rgb4[:, :, :3] = rgb[:, :, :]
        with sw.checkpoint("rgb2hsi3"):
            hsi4 = hsis.rgb2hsi3(rgb4)
        sw.showcheck(hsi4)
        with sw.checkpoint("hsi2rgb3"):
            rgb5 = hsis.hsi2rgb3(hsi4)
        sw.showcheck(rgb5)

    assert_closeenough(rgb2, rgb)
    assert_closeenough(rgb3, rgb)
    assert_closeenough(rgb5[:, :, :3], rgb)
    h3, s3, i3 = split_channels(hsi3)
    assert_closeenough(h3, h)
    assert_closeenough(s3, s)
    assert_closeenough(i3, i)


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
    with StopWatcher("xyz", kernel=cm) as sw:
        with sw.checkpoint("roundtrip"):
            xyz = cm.transform(rgb, 0, global_size=rgb.size // 3)
            rgb2 = cm.transform_named("adobe1998_d65.xyz2rgb", xyz)
        sw.showcheck(rgb2)


def test_xyz_view():
    cm = Colormath()
    rgb = image_test()
    with StopWatcher("srgb_d50", kernel=cm) as sw:
        with sw.checkpoint("xzy"):
            xyz = cm.transform_named("srgb_d50.rgb2xyz", rgb)
        sw.showcheck(xyz)
        x, y, z = split_channels(xyz)
        sw.showcheck(x, "X")
        sw.showcheck(y, "Y")
        sw.showcheck(z, "Z")
        ones = np.ones_like(x) * 0.3

        def checkshow(x1, y1, z1, checkpoint):
            a = join_channels(x1, y1, z1)
            with sw.checkpoint(checkpoint):
                rgbout = cm.transform_named("srgb_d50.xyz2rgb", xyz)
            sw.showcheck(rgbout, depth=1)

        checkshow(x, y, z, "2rgb")
        checkshow(ones, y, z, "1yz")
        checkshow(x, ones, z, "x1z")
        checkshow(x, y, ones, "xy1")


def test_colorspaces():
    cm = Colormath()
    rgb = image_test()
    transformed = []
    with StopWatcher("colormath", kernel=cm) as sw:
        sw.show(rgb, "rgb")
        with sw.checkpoint("srgb_d50.rgb2xyz"):
            xyz = cm.transform_named("srgb_d50.rgb2xyz", rgb)
        sw.showcheck(xyz)
        for name in transform_names[1::2]:
            with sw.checkpoint(name):
                rgb2 = cm.transform_named(name, xyz)
            sw.showcheck(rgb2)


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
    with StopWatcher("xyY", kernel=cm) as sw:
        with sw.checkpoint("xyz"):
            xyz = cm.transform_named("srgb_d50.rgb2xyz", rgb)
        sw.showcheck(xyz)
        with sw.checkpoint("xyY"):
            xyY = cm.xyz2xyY(xyz)
        sw.showcheck(xyY)
        with sw.checkpoint("xyz"):
            xyz2 = cm.xyY2xyz(xyY)
        sw.showcheck(xyz2)
        with sw.checkpoint("2rgb"):
            rgb2 = cm.transform_named("adobe1998_d65.xyz2rgb", xyz2)
        sw.showcheck(rgb2)
