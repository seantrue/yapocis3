"""
Higher level wrapper for anisotropic filter (Perona/Malik).
"""
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

# http://mail.scipy.org/pipermail/scipy-user/2009-February/020125.html
# Anisotropic Diffusion, as per Perona and Malik's paper (see section V).

import numpy

from yapocis.lib.channels import split_channels, join_channels
from yapocis.lib.hsi import *
from yapocis.lib.yapocis_types import *

from yapocis.lib.peronamalik import filterImage, filterImage_res

def _exp(image_gradient: Array, scale: float):
    return numpy.exp(-(numpy.absolute(image_gradient) / scale) ** 2)


def _inv(image_gradient: Array, scale: float):
    return 1 / (1 + (numpy.absolute(image_gradient) / scale) ** 2)

def pm_filter_n(image: Array, n: int, scale: float = 3.0, step_size: float = 0.2) -> Array:
    """
    Perform n iterations of peronamalik filter on 2D array.

    Args:
       image: 2d array, dtype = np.float32
       n: iterations
       scale: peronamalik scale parameter
       step_size: peronamalik step step parameter

    Returns:
       2D array containing filtered image
    """
    global peronamalik
    if step_size > 0.25:
        raise ValueError('step_size parameter must be <= 0.25 for numerical stability.')
    image = image.copy()
    if image.max() <= 1.0:
        print("Rescaling")
        image *= 256.
        image1 = peronamalik(image, scale, step_size) / 256.
        n -= 1
        while 1:
            if n <= 0:
                program.read(image1)
                return image1 / 256.
            filterImage_res(image1, scale, step_size, image)
            n -= 1
            if n <= 0:
                program.read(image)
                return image / 256.
            filterImage_res(image, scale, step_size, image1)
            n -= 1


def pm_filter(image: Array, scale: float = 3.0, step_size: float = 0.2) -> GeneratorExit:
    """
    Yields peronamalik filtered images from 2D array.

    Args:
       image: 2d array, dtype = np.float32
       scale: peronamalik scale parameter
       step_size: peronamalik step step parameter

    Returns:
      GeneratorExit
    """
    global peronamalik
    if step_size > 0.25:
        raise ValueError('step_size parameter must be <= 0.25 for numerical stability.')
    image = image.copy()
    if image.max() <= 1.0:
        print("Rescaling")
        image *= 256.
    while 1:
        image = filterImage(image, scale, step_size)
        yield image / 256


def pm_filter_rgb(rgb: Array, scale: float = 3.0, step_size: float = .2) -> GeneratorExit:
    """
    Yields peronamalik filtered RGB images from an WxHx3 array of RGB values.
    Filtering is done by extracting and filtering the intensity component.

    Args:
       image: 2d array, dtype = np.float32
       n: iterations
       scale: peronamalik scale parameter
       step_size: peronamalik step step parameter

    Returns:
       GeneratorExit
    """
    H, S, I = rgb2hsi(*split_channels(rgb))
    filtered = pm_filter(I, scale=scale, step_size=step_size)
    frame = 0
    while 1:
        fI = next(filtered)
        rgb = join_channels(*hsi2rgb(H, S, fI))
        frame += 1
        yield rgb


def pm_filter_rgbI(rgb: Array, n: int, scale: float = 3.0, step_size: float = .2) -> Array:
    """
    Perform n intereations of peronamalik filtering on a RGB image.
    Filtering is done by extracting and filtering the intensity component.

    Args:
       image: 3d array, dtype = np.float32
       n: iterations
       scale: peronamalik scale parameter
       step_size: peronamalik step step parameter

    Returns:
       RGB array containing filtered image
 """
    H, S, I = rgb2hsi(*split_channels(rgb))
    fI = pm_filter_n(I, n, scale=scale, step_size=step_size)
    rgb = join_channels(*hsi2rgb(H, S, fI))
    return rgb


def pm_filter_rgbIS(rgb: Array, scale: float = 3.0, step_size: float = .2) -> GeneratorExit:
    """
    Yields peronamalik filtered RGB images from an WxHx3 array of RGB values.
    Filtering is done by extracting and filtering the intensity component.

    Args:
       image: 3D array, dtype = np.float32
       scale: peronamalik scale parameter
       step_size: peronamalik step step parameter

    Returns:
       GeneratorExit
    """
    H, S, I = rgb2hsi(*split_channels(rgb))
    filteredI = pm_filter(I, scale=scale, step_size=step_size)
    filteredS = pm_filter(S, scale=scale, step_size=step_size)
    frame = 0
    while 1:
        fI = next(filteredI)
        fS = next(filteredS)
        rgb = join_channels(*hsi2rgb(H, fS, fI))
        frame += 1
        yield rgb
