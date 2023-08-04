"""
Wrapper for 2D gradient kernel
"""

#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

from yapocis.ocl.kernel_manager import load_program
from yapocis.utils.stereo_typing import *

from .bound_kernels import BoundKernels


class Gradient(BoundKernels):
    INTERFACE = "gradient"
