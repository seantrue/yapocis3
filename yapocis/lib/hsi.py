'''
Created on Jul 22, 2011

@author: seant
'''
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from yapocis.lib.bound_kernels import BoundKernels
from yapocis.utils.typing import *


class HSI(BoundKernels):
    INTERFACE="hsi"

    def rgb2hsi(self, r:Image1, g:Image1, b:Image1) -> Tuple[Image1,Image1,Image1]:
        """
        Convert red,green,blue planes to hue,saturation,intensity
    
        Args:
            r: red, a np.float32 array of 0-1.0 values
            g: green, ""
            b: blue, ""
    
        Returns:  hue, saturation, intensity as tuple of three arrays
    
        """
        h, s, i = self.kmgr.rgb2hsi(r, g, b)
        return h, s, i
    
    def hsi2rgb(self, h:Image1, s:Image1, i:Image1) -> Tuple[Array,Array,Array]:
        """
        Reverse rgb2hsi
    
        Args:
            h: hue, a np.float32 array of 0-1.0 values
            s: saturation, ""
            i: intensity, ""
    
        Returns:  red, green, blue tuple of three arrays
    
        """
        r, g, b = self.kmgr.hsi2rgb(h, s, i)
        return r,g,b
    
    def rgb2hsi3(self, rgb: Image3) -> Image3:
        """
        Convert rgb to hue,saturation,intensity
    
        Args:
            rgb: red,green,blue
    
        Returns:  hue, saturation, intensity as tuple of three arrays
    
        """
        hsi = self.kmgr.rgb2hsi3(rgb, global_size=rgb.size // rgb.shape[-1])
        return hsi
    
    def hsi2rgb3(self, hsi: Image3) -> Image3:
        """
        Reverse rgb2hsi3
    
        Args:
            hsi: hue, saturation, intensity Image3
        Returns: rgb Image3
    
        """
        rgb = self.kmgr.hsi2rgb3(hsi, global_size=hsi.size // hsi.shape[-1])
        return rgb
    
