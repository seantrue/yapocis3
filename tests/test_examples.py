#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from yapocis.lib.gradient import Gradient
from yapocis.lib.hsi import HSI
from yapocis.lib.channels import split_channels

from yapocis.utils import advanced
from yapocis.utils import basic

from . import image_test, snapshot, show_snapshots

def example_test(module, label):
    hsi = HSI()
    grad = Gradient(options=["-w"])
    image = image_test()[::2,::2,:]

    r, g, b = split_channels(image)
    h, s, i = hsi.rgb2hsi(r, g, b)
    g, a = grad.gradient(i, 7)
    module.show_array(f"Gradient ({label})", g)
    module.show_arraygrad(f"Grad angle ({label})", image, a)
    module.show_arraygrad(f"Grad vectors ({label})", image, a, g)
    snapshot(hsi)
    snapshot(grad)

def test_advanced_showarraygrad():
    example_test(advanced, "advanced")

def test_basic_showarraygrad():
    example_test(basic, "basic")
