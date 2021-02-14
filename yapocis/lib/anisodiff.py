"""
Higher level wrapper for anisotropic filter (Perona/Malik).
"""
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

# http://mail.scipy.org/pipermail/scipy-user/2009-February/020125.html
# Anisotropic Diffusion, as per Perona and Malik's paper (see section V).

from yapocis.lib.channels import split_channels, join_channels, join_planes, split_planes
from yapocis.lib.hsi import *
from yapocis.utils.typing import *

from yapocis.rpc.programs import import_program

filterImage = filterImage_res = filterImage2 = filterImage2_res = filterImage3 = filterImage3_res = filterImage4 = filterImage4_res = None
program, kernels = import_program("anisodiff", import_kernels="*")


def anisodiff_n(image: Image1234, n: int, scale: float = 3.0, step_size: float = 0.2) -> Image1234:
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
    if len(image.shape) == 2:
        filter = filterImage
        filter_res = filterImage_res
        global_size = a.size
    elif image.shape[2] == 2:
        filter = filterImage2
        filter_res = filterImage_res2
        global_size = a.size//2
    elif image.shape[2] == 3:
        filter = filterImage3
        filter_res = filterImage_res3
        global_size = a.size//3
    elif image.shape[2] == 4:
        filter = filterImage4
        filter_res = filterImage_res4
        global_size = a.size//4
    else:
        raise ValueError(f'Array shape {a.shape} is not supported')

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
            filter_res(image1, scale, step_size, image, global_size=global_size)
            n -= 1
            if n <= 0:
                program.read(image)
                return image / 256.
            filter_res(image, scale, step_size, image1, global_size=global_size)
            n -= 1


def anisodiff(image: Image1234, scale: float = 3.0, step_size: float = 0.2) -> Iterator[Image1234]:
    """
    Yields Perona/Malik filtered images from 2D array.

    Args:
       image: 2d array, dtype = np.float32
       scale: Perona/Malik scale
       step_size: Perona/Malik step

    Returns:
      GeneratorExit
    """
    if step_size > 0.25:
        raise ValueError('step_size parameter must be <= 0.25 for numerical stability.')
    if len(image.shape) == 2:
        filter = filterImage
        global_size = image.size
    elif image.shape[2] == 2:
        filter = filterImage2
        global_size = image.size//2
    elif image.shape[2] == 3:
        filter = filterImage3
        global_size = image.size//3
    elif image.shape[2] == 4:
        filter = filterImage4
        global_size = image.size//4
    else:
        raise ValueError(f'Array shape {a.shape} is not supported')
    image = image.copy()
    if image.max() <= 1.0:
        print("Rescaling")
        image *= 256.
    while 1:
        image = filter(image, scale, step_size, global_size=global_size)
        yield image / 256


def anisodiff_rgbI(rgb: Array, scale: float = 3.0, step_size: float = .2) -> Iterator[Image3]:
    """
    Yields Perona/Malik filtered RGB images from an WxHx3 array of RGB values.
    Filtering is done by extracting and filtering the intensity component.

    Args:
       image: 2d array, dtype = np.float32
       n: iterations
       scale: Perona/Malik scale parameter
       step_size: Perona/Malik step step parameter

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


def anisodiff_rgbI_n(rgb: Image3, n: int, scale: float = 3.0, step_size: float = .2) -> Image3:
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


def anisodiff_rgbIS(rgb: Image3, scale: float = 3.0, step_size: float = .2) -> Iterator[Image3]:
    """
    Yields Perona/Malik filtered RGB images from an WxHx3 array of RGB values.
    Filtering is done by extracting and filtering the intensity and saturation components.

    Args:
       image: 3D array, dtype = np.float32
       scale: Perona/Malik scale parameter
       step_size: Perona/Malik step step parameter

    Returns:
       sequence of 3D Arrays
    """
    H, S, I = rgb2hsi(*split_channels(rgb))
    SI = join_planes(S,I)
    filteredSI = anisodiff(SI, scale=scale, step_size=step_size)
    frame = 0
    while 1:
        fSI = next(filteredSI)
        fS, fI = split_planes(fSI)
        rgb = join_channels(*hsi2rgb(H, fS, fI))
        frame += 1
        yield rgb


def anisodiff_rgbIS_n(rgb: Image3, n: int, scale: float = 3.0, step_size: float = .2) -> Image3:
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
    SI = join_planes(S,I)
    fSI = anisodiff_n(SI, n, scale=scale, step_size=step_size)
    fS,fI = split_planes(fSI)
    rgb = join_channels(*hsi2rgb(H, fS, fI))
    return rgb
