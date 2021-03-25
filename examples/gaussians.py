'''
Created on Jul 21, 2011
@summary: Test yapocis lib1
@author: seant
'''

#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import numpy as np
from yapocis.utils.basic import imread
from yapocis.utils.basic import show_array, show_arraygrad
from yapocis.lib.median import Median
from yapocis.lib.gradient import Gradient
from yapocis.lib.hsi import HSI
from yapocis.lib.gaussian import Gaussian
from yapocis.lib.channels import split_channels, join_channels


def main():
    median = Median()
    gradient = Gradient()
    hsi = HSI()
    gaussian = Gaussian()
    # Load the test image and reduce dimensions
    image = imread("tests/test.jpg")
    image = image[::2, ::2, :]

    show_array("Original", image)
    # Add noise to the image
    image = image * .8 + np.random.uniform(0, .2, image.shape)
    show_array("Noisy", image)

    image = median.median3x3(image, 100)
    show_array("Filtered", image)

    r, g, b = split_channels(image)
    h, s, i = hsi.rgb2hsi(r, g, b)
    show_array("I", i)
    show_array("S", s)
    show_array("H", h)

    h, s, i = hsi.rgb2hsi(r, g, b)
    blur = gaussian.gauss_image(i, 3)
    show_array("Blur", blur)
    more_blurred = gaussian.gauss_image(i, 4)
    dog = blur - more_blurred
    show_array("DOG", dog)

    h, s, i = hsi.rgb2hsi(r, g, b)
    g, a = gradient.gradient(i, 5)
    show_array("Gradient", g)
    show_array("Angle", a)
    g, a = gradient.gradient(i, 5)
    sat = np.ones_like(i)
    gimg = join_channels(*hsi.hsi2rgb(a, sat, g))
    show_array("Color gradient with angle", gimg)
    g, a = gradient.gradient(i, 5)
    show_arraygrad("Grad angle", image, a)
    g, a = gradient.gradient(i, 5)
    show_arraygrad("Grad vectors", image, a, g * 10)


if __name__ == '__main__':
    main()
