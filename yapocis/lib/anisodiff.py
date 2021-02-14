"""
Higher level wrapper for anisotropic filter (Perona/Malik).
"""
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

# http://mail.scipy.org/pipermail/scipy-user/2009-February/020125.html
# Anisotropic Diffusion, as per Perona and Malik's paper (see section V).

from yapocis.lib.channels import split_channels, join_channels
from yapocis.lib.hsi import *
from yapocis.utils.typing import *

from yapocis.rpc.programs import import_program
filterImage = filterImage_res = filterImage4 = filterImage4_res = None
program, kernels= import_program("anisodiff", import_kernels="*")


def anisodiff_n(image: Array, n: int, scale: float = 3.0, step_size: float = 0.2) -> Array:
    """
    Perform n iterations of Perona/Malik filter on 2D/3D array.

    Args:
       image: 2d /3d array, dtype = np.float32
       n: iterations
       scale: Perona/Malik scale parameter
       step_size: Perona/Malik step parameter

    Returns:
       2D/3d array containing filtered image
    """
    if step_size > 0.25:
        raise ValueError('step_size parameter must be <= 0.25 for numerical stability.')
    filter = filterImage if len(image.shape) == 2 else filterImage4
    filter_res = filterImage_res if len(image.shape) == 2 else filterImage4_res
    global_size = a.size if len(image.shape) == 2 else a.size//4
    image = image.copy()
    if image.max() <= 1.0:
        print("Rescaling")
        image *= 256.
        image1 = filter(image, scale, step_size) / 256.
        n -= 1
        while 1:
            if n <= 0:
                program.read(image1)
                return image1 / 256.
            filter_res(image1, scale, step_size, image,global_size=global_size)
            n -= 1
            if n <= 0:
                program.read(image)
                return image / 256.
            filter_res(image, scale, step_size, image1,global_size=global_size)
            n -= 1


def anisodiff(image: Array, scale: float = 3.0, step_size: float = 0.2) -> GeneratorExit:
    """
    Yields Perona/Malik filtered images from 2D array.

    Args:
       image: 2d array, dtype = np.float32
       scale: peronamalik scale parameter
       step_size: peronamalik step step parameter

    Returns:
      GeneratorExit
    """
    if step_size > 0.25:
        raise ValueError('step_size parameter must be <= 0.25 for numerical stability.')
    filter = filterImage if len(image.shape) == 2 else filterImage4
    global_size = image.size if len(image.shape) == 2 else image.size//4
    image = image.copy()
    if image.max() <= 1.0:
        print("Rescaling")
        image *= 256.
    while 1:
        image = filter(image, scale, step_size,global_size=global_size)
        yield image / 256


def anisodiff_rgbI(rgb: Array, scale: float = 3.0, step_size: float = .2) -> GeneratorExit:
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
    filtered = anisodiff(I, scale=scale, step_size=step_size)
    frame = 0
    while 1:
        fI = next(filtered)
        rgb = join_channels(*hsi2rgb(H, S, fI))
        frame += 1
        yield rgb


def anisodiff_rgbI_n(rgb: Array, n: int, scale: float = 3.0, step_size: float = .2) -> Array:
    """
    Perform n intereations of Perona/Malik filtering on a RGB image.
    Filtering is done by extracting and filtering the intensity component.

    Args:
       image: 3d array, dtype = np.float32
       n: iterations
       scale: Perona/Malik scale parameter
       step_size: Perona/Malik step step parameter

    Returns:
       RGB array containing filtered image
 """
    H, S, I = rgb2hsi(*split_channels(rgb))
    fI = anisodiff_n(I, n, scale=scale, step_size=step_size)
    rgb = join_channels(*hsi2rgb(H, S, fI))
    return rgb


def anisodiff_rgbIS(rgb: Array, scale: float = 3.0, step_size: float = .2) -> GeneratorExit:
    """
    Yields Perona/Malik filtered RGB images from an WxHx3 array of RGB values.
    Filtering is done by extracting and filtering the intensity and saturation components.

    Args:
       image: 3D array, dtype = np.float32
       scale: Perona/Malik scale parameter
       step_size: Perona/Malik step step parameter

    Returns:
       GeneratorExit
    """
    H, S, I = rgb2hsi(*split_channels(rgb))
    filteredI = anisodiff(I, scale=scale, step_size=step_size)
    filteredS = anisodiff(S, scale=scale, step_size=step_size)
    frame = 0
    while 1:
        fI = next(filteredI)
        fS = next(filteredS)
        rgb = join_channels(*hsi2rgb(H, fS, fI))
        frame += 1
        yield rgb

def anisodiff_rgbIS_n(rgb: Array, n: int, scale: float = 3.0, step_size: float = .2) -> Array:
    """
    Perform n intereations of Perona/Malik filtering on a RGB image.
    Filtering is done by extracting and filtering the intensity and saturation components.

    Args:
       image: 3d array, dtype = np.float32
       n: iterations
       scale: Perona/Malik scale parameter
       step_size: Perona/Malik step step parameter

    Returns:
       RGB array containing filtered image
 """
    H, S, I = rgb2hsi(*split_channels(rgb))
    fI = anisodiff_n(I, n, scale=scale, step_size=step_size)
    fS = anisodiff_n(S, n, scale=scale, step_size=step_size)
    rgb = join_channels(*hsi2rgb(H, fS, fI))
    return rgb
