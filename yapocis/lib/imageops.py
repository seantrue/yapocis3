#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.
"""
Wrapper for 2D gradient kernel
"""

#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from yapocis.utils.stereo_typing import *

from .bound_kernels import BoundKernels


class Imageops(BoundKernels):
    INTERFACE = "imageops"
