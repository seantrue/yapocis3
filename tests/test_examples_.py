#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import numpy as np

from yapocis.utils.basic import imread
from yapocis.lib.gradient import gradient
from yapocis.lib.hsi import rgb2hsi
from yapocis.lib.channels import split_channels

from yapocis.utils.advanced import show_array, show_arraygrad
from . import test_image_path

test_image = test_image_path()

def test_advanced_showarraygrad():
    image = np.float32(imread(test_image))
    image /= 256.
    image = image[::2,::2,:]

    r, g, b = split_channels(image)
    h, s, i = rgb2hsi(r, g, b)
    g, a = gradient(i, 5)
    show_array("Gradient (advanced)", g)
    show_arraygrad("Grad angle (advanced)", image, a)
    show_arraygrad("Grad vectors (advanced)", image, a, g * 10)

