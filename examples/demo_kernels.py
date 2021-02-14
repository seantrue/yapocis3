'''
Created on Jul 21, 2011
@summary: Test yapocis lib
@author: seant
'''

#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import numpy as np
from yapocis.utils.basic import imread
from yapocis.utils.basic import show_array, show_arraygrad
from yapocis.lib.median import median3x3
from yapocis.lib.gradient import gradient
from yapocis.lib.hsi import rgb2hsi, hsi2rgb
from yapocis.lib.channels import split_channels, join_channels


def test():
    # Load the test image and reduce dimensions
    image = np.float32(imread("tests/test.jpg"))
    image /= 256.
    image_fs = image
    image = image_fs[::2,::2,:]

    show_array("Original", image)
    # Add noise to the image
    image = image *.8 + np.random.uniform(0,.2,image.shape)
    show_array("Noisy", image)

    image = median3x3(image, 100)
    show_array("Filtered", image)
    
    r,g,b = split_channels(image)
    h,s,i = rgb2hsi(r,g,b)
    show_array("I", i)
    show_array("S", s)
    show_array("H", h)

    from yapocis.lib.gaussian import gauss_image
    h,s,i = rgb2hsi(r,g,b)
    blur = gauss_image(i, 3)
    show_array("Blur", blur)
    blurmore = gauss_image(i, 4)
    dog = blur-blurmore
    show_array("DOG", dog)
    
    h,s,i = rgb2hsi(r,g,b)
    g,a = gradient(i,5)
    show_array("Gradient", g)
    show_array("Angle", a)
    g,a = gradient(i,5)
    sat = np.ones_like(i)
    gimg = join_channels(*hsi2rgb(a, sat, g))
    show_array("Color gradient with angle", gimg)
    g,a = gradient(i,5)
    show_arraygrad("Grad angle", image, a)
    g,a = gradient(i,5)
    show_arraygrad("Grad vectors", image, a, g * 10)
   
if __name__ == '__main__':
    test()
