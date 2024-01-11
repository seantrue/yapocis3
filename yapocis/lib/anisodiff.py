"""
Higher level wrapper for anisotropic filter (Perona/Malik).
"""
import numpy as np

#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

# http://mail.scipy.org/pipermail/scipy-user/2009-February/020125.html
# Anisotropic Diffusion, as per Perona and Malik's paper (see section V).

from yapocis.lib.channels import (join_channels, join_planes, split_channels,
                                  split_planes)
from yapocis.lib.hsi import *
from yapocis.utils.basic import get_image_depth
from yapocis.utils.stereo_typing import *

from .bound_kernels import BoundKernels
from .hsi import HSI


class Anisodiff(BoundKernels):
    INTERFACE = "anisodiff"
    filter_by_depth = ["filterImage", "filterImage2", "filterImage3", "filterImage4"]
    def __init__(self, device_name:Optional[str]=None, **kwargs):
        super().__init__(device_name=device_name,  **kwargs)
        self.hsi = HSI(device_name=device_name)
    def select_by_depth(self, image, suffix=""):
        depth = get_image_depth(image)
        attr = self.filter_by_depth[depth - 1]
        attr = f"{attr}{suffix}" if suffix else attr
        selected = getattr(self.kmgr, attr, None)
        if selected is None:
            raise ValueError(f'No {attr} kernel found for array shape {image.shape}')
        return selected

    def global_size_by_depth(self, image):
        depth = get_image_depth(image)
        return image.size // depth

    def anisodiff_n(self, image: Image1234, n: int, scale: float = 3.0, step_size: float = 0.2) -> Image1234:
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
        filter_res = self.select_by_depth(image, suffix="_res")
        global_size = self.global_size_by_depth(image)
        image = image.copy() if image.max() > 1.0 else image * 256.
        # Pad to make Cuda happy
        padding_required = len(image.shape) == 3 and image.shape[2] == 3
        if padding_required:
            zeros = np.zeros(image.shape[:2],dtype=image.dtype)
            input = np.dstack((image,zeros))
        else:
            input = image
        output = np.empty_like(input)
        filter_res.write(input)
        filter_res.write(output)
        for i in range(n):
            filter_res(input, scale, step_size, output, global_size=global_size, resident=(input, output))
            input, output = output, input
        filter_res.read(output)
        if padding_required:
            output = output[:,:,:3]
        return output / 256.

    def anisodiff(self, image: Image1234, scale: float = 3.0, step_size: float = 0.2) -> Iterator[Image1234]:
        """
        Yields Perona/Malik filtered images from 2D array.

        Args:
           image: 2d array, dtype = np.float32
           scale: Perona/Malik scale
           step_size: Perona/Malik step

        Returns:
          iterator of Image1234
        """
        if step_size > 0.25:
            raise ValueError('step_size parameter must be <= 0.25 for numerical stability.')
        filter_res = self.select_by_depth(image, suffix="_res")
        global_size = self.global_size_by_depth(image)
        image = image.copy()
        if image.max() <= 1.0:
            print("Rescaling")
            image *= 256.
        input = image
        output = np.empty_like(input)
        filter_res.write(input)
        filter_res.write(output)
        while True:
            filter_res(input, scale, step_size, output, global_size=global_size, resident=(input, output))
            filter_res.read(output)
            yield output / 256
            input, output = output, input

    def anisodiff_rgbI(self, rgb: Array, scale: float = 3.0, step_size: float = .2) -> Iterator[Image3]:
        """
        Yields Perona/Malik filtered RGB images from an WxHx3 array of RGB values.
        Filtering is done by extracting and filtering the intensity component.

        Args:
           rgb: 2d array, dtype = np.float32
           scale: Perona/Malik scale parameter
           step_size: Perona/Malik step step parameter

        Returns:
           iterator of Image3
        """
        H, S, I = self.hsi.rgb2hsi(*split_channels(rgb))
        filtered = self.anisodiff(I, scale=scale, step_size=step_size)

        frame = 0
        while 1:
            fI = next(filtered)
            rgb = join_channels(*self.hsi.hsi2rgb(H, S, fI))
            frame += 1
            yield rgb

    def anisodiff_rgbI_n(self, rgb: Image3, n: int, scale: float = 3.0, step_size: float = .2) -> Image3:

        """
                Perform n ç of Perona/Malik filtering on a RGB image.
            Filtering is done by extracting and filtering the intensity component.

            Args:
                   rgb: 3d array, dtype = np.float32
               n: iterations
               scale: Perona/Malik scale parameter
               step_size: Perona/Malik step step parameter

            Returns:
               RGB array containing filtered image
         """
        H, S, I = self.hsi.rgb2hsi(*split_channels(rgb))
        fI = self.anisodiff_n(I, n, scale=scale, step_size=step_size)
        rgb = join_channels(*self.hsi.hsi2rgb(H, S, fI))
        return rgb

    def anisodiff_rgbIS(self, rgb: Image3, scale: float = 3.0, step_size: float = .2) -> Iterator[Image3]:
        """
        Yields Perona/Malik filtered RGB images from an WxHx3 array of RGB values.
        Filtering is done by extracting and filtering the intensity and saturation components.

        Args:
               rgb: 3D array, dtype = np.float32
           scale: Perona/Malik scale parameter
           step_size: Perona/Malik step step parameter

        Returns:
           sequence of 3D Arrays
        """
        H, S, I = self.hsi.rgb2hsi(*split_channels(rgb))
        SI = join_planes(S, I)
        filteredSI = self.anisodiff(SI, scale=scale, step_size=step_size)

        frame = 0
        while 1:
            fSI = next(filteredSI)
            fS, fI = split_planes(fSI)
            rgb = join_channels(*self.hsi.hsi2rgb(H, fS, fI))
            frame += 1
            yield rgb

    def anisodiff_rgbIS_n(self, rgb: Image3, n: int, scale: float = 3.0, step_size: float = .2) -> Image3:

        """
            Perform n iterations of Perona/Malik filtering on a RGB image.
        Filtering is done by extracting and filtering the intensity and saturation components.

        Args:
               rgb: 3d array, dtype = np.float32
           n: iterations
           scale: Perona/Malik scale parameter
           step_size: Perona/Malik step step parameter

        Returns:
           RGB array containing filtered image
        """
        H, S, I = self.hsi.rgb2hsi(*split_channels(rgb))
        SI = join_planes(S, I)
        fSI = self.anisodiff_n(SI, n, scale=scale, step_size=step_size)
        fS, fI = split_planes(fSI)
        rgb = join_channels(*self.hsi.hsi2rgb(H, fS, fI))
        return rgb
