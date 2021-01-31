import numpy as np

from yapocis.utils import imread
from yapocis.lib.median import median3x3
from yapocis.lib.gradient import gradient
from yapocis.lib.hsi import rgb2hsi, hsi2rgb
from yapocis.channels import split_channels, join_channels

from yapocis.advanced import show_array, show_arraygrad
from . import test_image_path

test_image = test_image_path()

def test_advanced_showarraygrad():
    image = np.float32(imread(test_image))
    image /= 256.

    r, g, b = split_channels(image)
    h, s, i = rgb2hsi(r, g, b)
    g, a = gradient(i, 5)
    show_array("Gradient", g)
    show_arraygrad("Grad angle", image, a)
    show_arraygrad("Grad vectors", image, a, g * 10)

