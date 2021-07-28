"""
Wrapper for 2D gradient kernel
"""

#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from yapocis.ocl.kernel_manager import load_program
from yapocis.utils.stereo_typing import *

from .bound_kernels import BoundKernels


class Gradient(BoundKernels):
    INTERFACE = "gradient"
