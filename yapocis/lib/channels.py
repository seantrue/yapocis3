"""
Accommodate the differences between PIL, Numpy, and convenient programming models for OpenCL
HWC <-> CHW ordering, conceptually.
"""

#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import numpy as np

from yapocis.utils.stereo_typing import *

def histeq(im:Image1, nbr_bins:int =2 ** 16) -> Image1:
    """
    Histogram equalization of a 2D array

    Args:
        im: 2D Array
        nbr_bins: number of bins

    Returns:
        2d Array, dtype=np.float32
    """
    imhist, bins = np.histogram(im.ravel(), nbr_bins, normed=True)
    imhist[0] = 0

    cdf = imhist.cumsum() #cumulative distribution function
    cdf = (nbr_bins - 1) * cdf / cdf[-1]
    cdf = cdf / nbr_bins

    im2 = np.interp(im.flatten(), bins[:-1], cdf)
    return np.array(im2, np.float32).reshape(im.shape)


def split_channels(rgb:Image3) -> Tuple[Image1,Image1,Image1]:
    """
    Split a WxHx3 array into WxH, WxH, WxH

    Args:
        rgb: 3D array

    Returns:
        Tuple of 2D arrays that are copies of the original data
    """
    channels =  np.dsplit(rgb, indices_or_sections=3)
    return [np.ascontiguousarray(np.squeeze(channel)) for channel in channels]

def join_channels(r: Image1, g: Image1, b: Image1) -> Image3:
    """
    Join WxH, WxH, WxH 2D arrays into a 3D Array with copied data

    Args:
        r: WxH array
        g: WxH array
        b: WxH array

    Returns:
        3xWxH array

    """
    rgb = np.ascontiguousarray(np.dstack((r,g,b)))
    return rgb

def split_planes(image: Image1234) -> Tuple:
    if len(image.shape) == 2:
        return image.copy()
    planes = [np.ascontiguousarray(np.squeeze(plane)) for plane in np.dsplit(image, indices_or_sections=image.shape[-1])]
    return planes

def join_planes(*planes) -> Image1234:
    image = np.ascontiguousarray(np.dstack(planes))
    return image


