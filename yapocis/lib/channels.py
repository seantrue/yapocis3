#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import numpy as np

from yapocis.lib.yapocis_types import Tuple
from yapocis.lib.yapocis_types import Array


def histeq(im:Array, nbr_bins:int =2 ** 16) -> Array:
    """
    Histogram equalization of a 2D array
    Args:
        im: 2D Array
        nbr_bins: number of bins

    Returns:
        2d Array, dtype=np.float32
    """
    imhist, bins = np.histogram(im.flatten(), nbr_bins, normed=True)
    imhist[0] = 0

    cdf = imhist.cumsum() #cumulative distribution function
    cdf ** .5
    cdf = (nbr_bins - 1) * cdf / cdf[-1]
    cdf = cdf / (nbr_bins)

    im2 = np.interp(im.flatten(), bins[:-1], cdf)
    return np.array(im2, np.float32).reshape(im.shape)


def split_channels(rgb:Array) -> Tuple[Array,Array,Array]:
    """
    Split a 3xWxH array into WxH, WxH, WxH
    Args:
        rgb: 3D array

    Returns:
        Tuple of 2D arrays that are copies of the original data
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return r.copy(), g.copy(), b.copy()


def join_channels(r: Array, g: Array, b: Array) -> Array:
    """
    Join WxH, WxH, WxH 2D arrays into a 3D Array with copied data
    Args:
        r: WxH array
        g: WxH array
        b: WxH array

    Returns:
        3xWxH array

    """
    shape = list(r.shape)
    shape.append(3)
    rgb = np.empty(tuple(shape), dtype=r.dtype)
    rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2] = r.copy(), g.copy(), b.copy()
    return rgb