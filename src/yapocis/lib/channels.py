"""
Accommodate the differences between PIL, Numpy, and convenient programming models for OpenCL
HWC <-> CHW ordering, conceptually.
"""

#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

from yapocis.utils.stereo_typing import *


def split_channels(rgb: Image3) -> tuple[Image1, Image1, Image1]:
    """
    Split a WxHx3 array into WxH, WxH, WxH

    Args:
        rgb: 3D array

    Returns:
        tuple of 2D arrays that are copies of the original data
    """
    channels = np.dsplit(rgb, indices_or_sections=3)
    return tuple(np.ascontiguousarray(np.squeeze(channel)) for channel in channels)


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
    rgb = np.ascontiguousarray(np.dstack((r, g, b)))
    return rgb


def split_planes(image: Image1234) -> tuple:
    if len(image.shape) == 2:
        return image.copy()
    planes = [
        np.ascontiguousarray(np.squeeze(plane))
        for plane in np.dsplit(image, indices_or_sections=image.shape[-1])
    ]
    return planes


def join_planes(*planes) -> Image1234:
    image = np.ascontiguousarray(np.dstack(planes))
    return image
