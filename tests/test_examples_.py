#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from yapocis.lib.gradient import Gradient
from yapocis.lib.hsi import rgb2hsi
from yapocis.lib.channels import split_channels

from yapocis.utils.advanced import show_array, show_arraygrad
from . import image_test


def test_advanced_showarraygrad():
    grad = Gradient()
    image = image_test()
    image = image[::2,::2,:]

    r, g, b = split_channels(image)
    h, s, i = rgb2hsi(r, g, b)
    g, a = grad.gradient(i, 5)
    show_array("Gradient (advanced)", g)
    show_arraygrad("Grad angle (advanced)", image, a)
    show_arraygrad("Grad vectors (advanced)", image, a, g * 10)

