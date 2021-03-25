#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.
"""
Wrapper for 2D gradient kernel
"""

#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import numpy as np
from .bound_kernels import BoundKernels


class Imageops(BoundKernels):
    INTERFACE = "imageops"

    def histogram(self, input, n_bins=256, n_workers=32, low=0.0, high=1.0):
        partials = np.zeros((n_workers, n_bins), dtype=np.int32)
        local_size = input.size // n_workers
        self.kmgr.histogram(input, low, high, partials, local_size, global_size=n_workers)
        h = np.sum(partials, axis=0)
        del partials
        return h

    def histogram_res(self, input, partials, low=0.0, high=1.0):
        n_workers, n_bins = partials.shape
        local_size = input.size // n_workers
        self.kmgr.histogram_res(input, low, high, partials, local_size, global_size=n_workers)
