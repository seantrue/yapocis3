'''
Created on Jul 22, 2011

@author: seant
'''
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from yapocis.rpc.programs import import_program

from yapocis.utils.typing import *

rgb2hsi_cl = hsi2rgb_cl = rgb2hsi3_cl =hsi2rgb3_cl = None
program, kernels = import_program("hsi",
                                  rgb2hsi="rgb2hsi_cl", hsi2rgb="hsi2rgb_cl",
                                  rgb2hsi3="rgb2hsi3_cl", hsi2rgb3="hsi2rgb3_cl" )

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

def rgb2hsi3(rgb: Image3) -> Image3:
    """
    Convert rgb to hue,saturation,intensity

    Args:
        rgb: red,
        g: green, ""
        b: blue, ""

    Returns:  hue, saturation, intensity as tuple of three arrays

    """
    hsi = rgb2hsi3_cl(rgb, global_size=rgb.size//rgb.shape[-1])
    return hsi

def hsi2rgb3(hsi: Image3) -> Image3:
    """
    Reverse rgb2hsi3

    Args:
        hsi: hue, saturation, intensity Image3
    Returns: rgb Image3

    """
    rgb = hsi2rgb3_cl(hsi, global_size=hsi.size//hsi.shape[-1])
    return rgb

