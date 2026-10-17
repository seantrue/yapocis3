'''
gaussians.py

This script is part of the YapociS project, which is a Python wrapper implementing
general purpose computing on graphics processing units (GPUs) with OpenCL.
Specifically, this file uses the library to perform a variety of operations on images.

Imports:
    numpy - A fundamental package for numerical computations in Python.
    join_channels, split_channels, Gaussian, Gradient, HSI, Median - These are a variety of functions for managing images,
                                                                    including channel management, applying Gaussian blur,
                                                                    creating gradients, converting color modes, and applying median filtering.
    imread, show_array, show_arraygrad - These are utility functions used for reading images (imread) and visualizing them (show_array and show_arraygrad).

The main function:
    Initializes instances of the Median, Gradient, HSI, and Gaussian classes.
    Loads a test image and performs preprocessing. It displays the original and noised versions of the image.
    Applies a median filter to the noisy image and displays the result.
    Splits the image into its constituent color channels, converts it to the HSI color space, and individually displays each channel (Hue, Saturation, Intensity).
    Applies two levels of blurring to the image with a Gaussian function and performs a Difference of Gaussians (DoG) computation between them, after which it displays the result.
    Calculates the gradient and gradient angle of the image and displays both.
    From the HSI color space representation, it produces a color gradient image and the corresponding gradient arrays.
    Calculates gradient vectors and visualizes them.

When this file is executed as a standalone script (not imported as a module), the main function is run.

Created on Jul 21, 2011
@summary: Test yapocis lib1
@author: seant
'''

#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.

import numpy as np

from yapocis.lib.channels import join_channels, split_channels
from yapocis.lib.gaussian import Gaussian
from yapocis.lib.gradient import Gradient
from yapocis.lib.hsi import HSI
from yapocis.lib.median import Median
from yapocis.utils.basic import show_array, show_arraygrad, get_test_image


def main():
    median = Median()
    gradient = Gradient()
    hsi = HSI()
    gaussian = Gaussian()
    # Load the test image and reduce dimensions
    image = get_test_image()
    image = image[::2, ::2, :]

    show_array(image, "Original")
    # Add noise to the image
    noisy = image * .8 + np.random.uniform(0, .2, image.shape)
    show_array(noisy, "Noisy")
    r, g, b = split_channels(noisy)
    r, g, b = median.median3x3(r, 100), median.median3x3(g, 100), median.median3x3(b, 100)
    filtered = join_channels(r, g, b)
    show_array(filtered, "Filtered")

    h, s, i = hsi.rgb2hsi(r, g, b)
    show_array(i, "I")
    show_array(s, "S")
    show_array(h, "H")

    h, s, i = hsi.rgb2hsi(r, g, b)
    blur = gaussian.gauss_image(i, 3)
    show_array(blur, "Blur")
    more_blurred = gaussian.gauss_image(i, 4)
    dog = blur - more_blurred
    show_array("DOG", dog)

    h, s, i = hsi.rgb2hsi(r, g, b)
    g, a = gradient.gradient(i, 5)
    show_array(g, "Gradient", )
    show_array(a, "Angle")
    g, a = gradient.gradient(i, 5)
    sat = np.ones_like(i)
    gimg = join_channels(*hsi.hsi2rgb(a, sat, g))
    show_array(gimg, "Color gradient with angle")
    g, a = gradient.gradient(i, 5)
    show_arraygrad("Grad angle", image, a)
    g, a = gradient.gradient(i, 5)
    show_arraygrad("Grad vectors", image, a, g * 10)


if __name__ == '__main__':
    main()
