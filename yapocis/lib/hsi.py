#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from yapocis.lib.bound_kernels import BoundKernels
from yapocis.utils.stereo_typing import *


class HSI(BoundKernels):
    INTERFACE="hsi"

    def rgb2hsi3(self, rgb: Image3) -> Image3:
        """
        Convert rgb to hue,saturation,intensity
    
        Args:
            rgb: red,green,blue
    
        Returns:  hue, saturation, intensity as tuple of three arrays
    
        """
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
        rgb = self.kmgr.hsi2rgb3(hsi, global_size=hsi.size // hsi.shape[-1])
        rgb = rgb.copy()
        self.purge()
        return rgb
