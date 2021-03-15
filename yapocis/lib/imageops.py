#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.
"""
Wrapper for 2D gradient kernel
"""

#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from yapocis.utils.typing import *

from .bound_kernels import BoundKernels


class Imageops(BoundKernels):
    INTERFACE = "imageops"

    def clip(self, input: Image1, low: np.float32, high: np.float32) -> Image1:
        return self.kmgr.clip(input, low, high)

    def stretch(self, input: Image1, input_min: np.float32, input_max: np.float32, low: np.float32,
                high: np.float32) -> Image1:
        return self.kmgr.stretch(input, input_min, input_max, low, high)

    def binarize(self, input: Image1, threshold: np.float32) -> Image1:
        return self.kmgr.binarize(input, threshold)
