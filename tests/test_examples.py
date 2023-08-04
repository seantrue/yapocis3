#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

from yapocis.lib.channels import split_channels
from yapocis.lib.gradient import Gradient
from yapocis.lib.hsi import HSI
from yapocis.utils.advanced import show_array
from yapocis.utils.advanced import show_array as show_arrayb
from yapocis.utils.advanced import show_arraygrad
from yapocis.utils.advanced import show_arraygrad as show_arraygradb

from . import image_test, show_snapshots, snapshot


def test_advanced_showarraygrad():
    hsi = HSI()
    grad = Gradient(options=["-w"])
    image = image_test()[::2,::2,:]

    r, g, b = split_channels(image)
    h, s, i = hsi.rgb2hsi(r, g, b)
    g, a = grad.gradient(i, 7)
    show_array("Gradient (advanced)", g)
    show_arraygrad("Grad angle (advanced)", image, a)
    show_arraygrad("Grad vectors (advanced)", image, a, g)
    snapshot(hsi)
    snapshot(grad)

def test_basic_showarraygrad():
    hsi = HSI()
    grad = Gradient(options=["-w"])
    image = image_test()[::2,::2,:]

    r, g, b = split_channels(image)
    h, s, i = hsi.rgb2hsi(r, g, b)
    g, a = grad.gradient(i, 7)
    show_arrayb("Gradient (advanced)", g)
    show_arraygradb("Grad angle (advanced)", image, a)
    show_arraygradb("Grad vectors (advanced)", image, a, g)
    snapshot(hsi)
    snapshot(grad)
