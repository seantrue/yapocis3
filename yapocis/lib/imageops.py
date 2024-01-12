#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.
"""
Wrapper for 2D gradient kernel
"""

#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

import numpy as np
from .bound_kernels import BoundKernels


class Imageops(BoundKernels):
    INTERFACE = "imageops"

    def histogram(self, input, n_bins=256, n_workers=32, low=0.0, high=1.0):
        hst = np.zeros(n_bins, dtype=np.int32)
        self.kmgr.histogram(input, low, high, hst)
        return hst

    def histogram_res(self, input, hst, low=0.0, high=1.0):
        self.kmgr.histogram_res(input, low, high, hst)
        return hst
