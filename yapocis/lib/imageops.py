#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.
"""
Wrapper for 2D gradient kernel
"""

#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

from yapocis.utils.stereo_typing import *

from .bound_kernels import BoundKernels


class Imageops(BoundKernels):
    INTERFACE = "imageops"
