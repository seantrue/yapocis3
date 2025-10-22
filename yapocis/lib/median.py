'''
Provides 2d median filter for 2d arrays
'''
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

# Provision this program with just one median filter, fixed length of 9
import os
from yapocis.utils.stereo_typing import *

from .bound_kernels import BoundKernels


class Median(BoundKernels):
    INTERFACE="median3x3"
    def __init__(self, device_name: str | None = None, use_package: bool | None = None):
        super().__init__(device_name=device_name, width=9, steps=[9])

        # Determine whether to use compiled kernel package
        if use_package is None:
            use_package = os.environ.get('YAPOCIS_KERNELPACK', '').lower() in ('1', 'true', 'yes')

        # Attempt to use package if requested
        self.use_package = False
        if use_package:
            try:
                self._init_with_package()
                self.use_package = True
            except (ImportError, AttributeError):
                # Package not available, already using traditional path
                pass

    def _init_with_package(self):
        """Initialize using compiled kernel package (fast path)."""
        from yapocis.kernelpack import get_kernelpack

        # Get compiled package for current backend
        self.package = get_kernelpack('median3x3', self.kmgr)
        self.package.load()

    def median3x3slow(self, image: Image1, iterations: int = 1) -> Image1:
        """
        3x3 median filter reading and writing the data on each pass

        Args:
            image:
            iterations:

        Returns: A copy of the array.

        """
        if self.use_package:
            while iterations > 0:
                image = self.package.median3x3(image)
                iterations -= 1
            return image.copy()
        else:
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

        if self.use_package:
            # Package path: Use kmgr for buffer ops, package for kernels
            input = image
            output = np.empty_like(input)
            self.kmgr.write(input)
            self.kmgr.write(output)
            while True:
                self.package.median3x3_res(input, output)
                iterations -= 1
                if iterations <= 0:
                    self.kmgr.read(output)
                    return output.copy()
                input, output = output, input
        else:
            # Traditional path
            m3 = self.kmgr.median3x3_res
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
