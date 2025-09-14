"""
Wrapper for 2D gradient kernel
"""

#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

from yapocis.utils.stereo_typing import *

from .bound_kernels import BoundKernels


class Gradient(BoundKernels):
    INTERFACE = "gradient"
    def gradient(self, a, stride=3):
        return self.kmgr.gradient(a, stride)
