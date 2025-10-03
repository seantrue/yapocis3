#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

from yapocis.lib.channels import split_channels
from yapocis.lib.gradient import Gradient
from yapocis.lib.hsi import HSI
from yapocis.utils import basic, advanced

from . import image_test, show_snapshots, snapshot

def example_test(module, label):
    hsi = HSI()
    grad = Gradient(options=["-w"])
    image = image_test()[::2,::2,:]

    r, g, b = split_channels(image)
    h, s, i = hsi.rgb2hsi(r, g, b)
    g, a = grad.gradient(i, 7)
    module.show_array(g, f"Gradient ({label})")
    module.show_arraygrad(f"Grad angle ({label})", image, a)
    module.show_arraygrad(f"Grad vectors ({label})", image, a, g)
    snapshot(hsi)
    snapshot(grad)

def test_advanced_showarraygrad():
    example_test(advanced, "advanced")

def test_basic_showarraygrad():
    example_test(basic, "basic")
