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


class Anisodiff:
    def __init__(self, device_name: str = None):
        self.kmgr = load_program("anisodiff", engine_name=device_name)
        self.filter_by_depth= ["filterImage","filterImage2", "filterImage3", "filterImage4"]

    def get_image_depth(self, image):
        if len(image.shape) in (2, 3):
            if len(image.shape) == 2:
                return 1
            elif image.shape[2] == 1:
                return 1
            elif image.shape[2] == 2:
                return 2
            elif image.shape[2] == 3:
                return 3
            elif image.shape[2] == 4:
                return 4
        raise ValueError(f'Array shape {image.shape} is not supported')

    def select_by_depth(self, image, suffix=""):
        depth = self.get_image_depth(image)
        attr = self.filter_by_depth[depth-1]
        selected = getattr(self.kmgr, attr, None)
        if selected is None:
            raise ValueError(f'No {attr} kernel found for array shape {image.shape}')
        return selected

    def global_size_by_depth(self, image):
        depth = self.get_image_depth(image)
        return image.size//depth

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
        filtr = self.select_by_depth(image)
        filter_res = self.select_by_depth(image,suffix="_res")
        global_size = self.global_size_by_depth(image)
        image = image.copy()
        if image.max() <= 1.0:
            print("Rescaling")
            image *= 256.
        image1 = filtr(image, scale, step_size) / 256.
        n -= 1
        while 1:
            if n <= 0:
                self.kmgr.read(image1)
                return image1 / 256.
            filter_res(image1, scale, step_size, image, global_size=global_size)
            n -= 1
            if n <= 0:
                self.kmgr.read(image)
                return image / 256.
            filter_res(image, scale, step_size, image1, global_size=global_size)
            n -= 1

    def anisodiff(self, image: Image1234, scale: float = 3.0, step_size: float = 0.2) -> Iterator[Image1234]:
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
        filtr = self.select_by_depth(image)
        global_size = self.global_size_by_depth(image)
        image = image.copy()
        if image.max() <= 1.0:
            print("Rescaling")
            image *= 256.
        while 1:
            image = filtr(image, scale, step_size, global_size=global_size)
            yield image / 256

    def anisodiff_rgbI(self, rgb: Array, scale: float = 3.0, step_size: float = .2) -> Iterator[Image3]:
        """
        Yields Perona/Malik filtered RGB images from an WxHx3 array of RGB values.
        Filtering is done by extracting and filtering the intensity component.

        Args:
               rgb: 2d array, dtype = np.float32
           scale: Perona/Malik scale parameter
           step_size: Perona/Malik step step parameter

        Returns:
           GeneratorExit
        """
        H, S, I = rgb2hsi(*split_channels(rgb))
        filtered = self.anisodiff(I, scale=scale, step_size=step_size)

        frame = 0
        while 1:
            fI = next(filtered)
            rgb = join_channels(*hsi2rgb(H, S, fI))
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
        H, S, I = rgb2hsi(*split_channels(rgb))
        fI = self.anisodiff_n(I, n, scale=scale, step_size=step_size)
        rgb = join_channels(*hsi2rgb(H, S, fI))
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
        H, S, I = rgb2hsi(*split_channels(rgb))
        SI = join_planes(S, I)
        filteredSI = self.anisodiff(SI, scale=scale, step_size=step_size)

        frame = 0
        while 1:
            fSI = next(filteredSI)
            fS, fI = split_planes(fSI)
            rgb = join_channels(*hsi2rgb(H, fS, fI))
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
        H, S, I = rgb2hsi(*split_channels(rgb))
        SI = join_planes(S, I)
        fSI = self.anisodiff_n(SI, n, scale=scale, step_size=step_size)
        fS, fI = split_planes(fSI)
        rgb = join_channels(*hsi2rgb(H, fS, fI))
        return rgb
