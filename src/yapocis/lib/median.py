"""
Provides 2d median filter for 2d arrays
"""

#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

# Provision this program with just one median filter, fixed length of 9
from yapocis.utils.stereo_typing import *

from .bound_kernels import BoundKernels


class Median(BoundKernels):
    INTERFACE = "median3x3"
    use_async = None

    def median3x3slow(self, image: Image1, iterations: int = 1) -> Image1:
        """
        3x3 median filter

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
        if iterations == 0:
            return image.copy()

        m3 = self.kmgr.median3x3_res
        input = image
        output = np.zeros_like(input)
        m3.write(input)
        m3.write(output)
        result_buffer = None
        if self.use_async:
            # Async path: enqueue all iterations without waiting
            with self.kmgr.async_mode():
                for i in range(iterations):
                    m3(input, output, resident=(input, output))
                    result_buffer = output
                    input, output = output, input
        else:
            # Synchronous path (traditional)
            for i in range(iterations):
                m3(input, output)
                result_buffer = output
                input, output = output, input

        m3.read(result_buffer)
        return result_buffer.copy()

    median3x3 = median3x3fast
