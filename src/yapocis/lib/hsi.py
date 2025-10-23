#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

import os
from yapocis.lib.bound_kernels import BoundKernels
from yapocis.utils.stereo_typing import *


class HSI(BoundKernels):
    INTERFACE = "hsi"

    def __init__(self, *args, use_package: bool | None = None, **kwargs):
        super().__init__(*args, **kwargs)

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
        from yapocis.kernelpack import get_kernelpack, get_kernel_names

        # Get compiled package for current backend
        self.package = get_kernelpack('hsi', self.kmgr)
        kernel_names = get_kernel_names('hsi')
        self.package.load(kernel_names)

    def rgb2hsi3(self, rgb: Image3) -> Image3:
        """
        Convert rgb to hue,saturation,intensity

        Args:
            rgb: red,green,blue

        Returns:  hue, saturation, intensity as tuple of three arrays

        """
        if self.use_package:
            hsi = self.package.rgb2hsi3(rgb)
        else:
            hsi = self.kmgr.rgb2hsi3(rgb, global_size=rgb.size // rgb.shape[-1])
        hsi = hsi.copy()
        self.purge()
        return hsi

    def hsi2rgb3(self, hsi: Image3) -> Image3:
        """
        Reverse rgb2hsi3

        Args:
            hsi: hue, saturation, intensity Image3
        Returns: rgb Image3

        """
        if self.use_package:
            rgb = self.package.hsi2rgb3(hsi)
        else:
            rgb = self.kmgr.hsi2rgb3(hsi, global_size=hsi.size // hsi.shape[-1])
        rgb = rgb.copy()
        self.purge()
        return rgb
