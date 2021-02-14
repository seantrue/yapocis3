'''
Created on Jul 22, 2011

@author: seant
'''
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import yapocis.rpc.programs

from yapocis.utils.typing import *

rgb2hsi_cl = hsi2rgb_cl = None
program, kernels = yapocis.rpc.programs.import_program("hsi", rgb2hsi="rgb2hsi_cl", hsi2rgb="hsi2rgb_cl")

def rgb2hsi(r:Image1, g:Image1, b:Image1) -> Tuple[Image1,Image1,Image1]:
    """
    Convert red,green,blue planes to hue,saturation,intensity

    Args:
        r: red, a np.float32 array of 0-1.0 values
        g: green, ""
        b: blue, ""

    Returns:  hue, saturation, intensity as tuple of three arrays

    """
    h, s, i = rgb2hsi_cl(r, g, b)
    return h, s, i

def hsi2rgb(h:Image1, s:Image1, i:Image1) -> Tuple[Array,Array,Array]:
    """
    Reverse rgb2hsi

    Args:
        h: hue, a np.float32 array of 0-1.0 values
        s: saturation, ""
        i: intensity, ""

    Returns:  red, green, blue tuple of three arrays

    """
    r,g,b = hsi2rgb_cl(h,s,i)
    return r,g,b


