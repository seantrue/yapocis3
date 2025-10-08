'''
Provides 2d median filter for 2d arrays
'''
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

# Provision this program with just one median filter, fixed length of 9
from yapocis.utils.stereo_typing import *

from .bound_kernels import BoundKernels


class Median(BoundKernels):
    INTERFACE="median3x3"
    def __init__(self, device_name: str | None = None):
        super().__init__(device_name=device_name, width=9, steps=[9])

    def median3x3slow(self, image: Image1, iterations: int = 1) -> Image1:
        """
        3x3 median filter reading and writing the data on each pass

        Args:
            image:
            iterations:

        Returns: A copy of the array.

        """
        while iterations > 0:
            image = self.kmgr.median3x3(image)
            iterations -= 1
        return image.copy()

    def median3x3fast(self, image: Image1, iterations: int = 1) -> Image1:
        """
        3x3 median filter iterating use ping/pong buffers on the device.

        Args:
            image: 2D Array
            iterations:

        Returns: A copy of the array.

        """
        m3 = self.kmgr.median3x3_res
        if iterations == 0:
            return image.copy()
        input = image
        output = np.empty_like(input)
        m3.write(input)
        m3.write(output)
        while True:
            m3(input, output, resident=(input, output))
            iterations -= 1
            if iterations <= 0:
                m3.read(output)
                return output.copy()
            input, output = output, input

    median3x3 = median3x3fast
