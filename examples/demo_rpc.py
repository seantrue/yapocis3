'''
Created on Jul 21, 2011
@summary: Test yapocis kernels
@author: seant
'''

import numpy as np
from yapocis.utils import imread
from yapocis.utils import show_array, show_arraygrad
from yapocis.kernels.median import median3x3
from yapocis.kernels.gradient import gradient
from yapocis.kernels.hsi import rgb2hsi, hsi2rgb
from yapocis.channels import split_channels, join_channels


def test():

    # Create a noisy image with an embedded white square
    image = np.zeros((201,199),dtype=np.float32)
    width,height = image.shape
    x,y = width//2, height//2
    offset = 10
    image[x-offset:x+offset,y-offset:y+offset] = 2
    image += np.random.random_sample(image.shape)
    
    filtered = median3x3(image, 100)

    show_array("Noisy", image)
    show_array("Filtered", filtered)
    
    image = np.float32(imread("../yapocis/test.jpg"))
    image /= 256.

    show_array("Test HSI", image)
    r,g,b = split_channels(image)
    h,s,i = rgb2hsi(r,g,b)
    show_array("I", i)
    show_array("S", s)
    show_array("H", h)

    from yapocis.kernels.gaussian import gauss_image
    blur = gauss_image(i, 3)
    show_array("Blur", blur)
    blurmore = gauss_image(i, 4)
    dog = blur-blurmore
    show_array("DOG", dog)
    
    g,a = gradient(i,5)
    show_array("Gradient", g)
    show_array("Angle", a)
    sat = np.ones_like(i)
    gimg = join_channels(*hsi2rgb(a, sat, g))
    show_array("Color gradient with angle", gimg)
    show_arraygrad("Grad angle", image, a)
    show_arraygrad("Grad vectors", image, a, g * 10)
   
if __name__ == '__main__':
    test()
