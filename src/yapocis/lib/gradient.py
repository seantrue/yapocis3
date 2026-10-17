"""
Wrapper for 2D gradient kernel
"""

#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.

from yapocis.utils.stereo_typing import *

from .bound_kernels import BoundKernels


class Gradient(BoundKernels):
    INTERFACE = "gradient"

    def gradient(self, a, stride=3):
        return self.kmgr.gradient(a, stride)
