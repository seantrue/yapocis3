"""
Low level wrapper for zero crossing detector kernel
"""

#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

from .bound_kernels import BoundKernels


class ZC(BoundKernels):
    INTERFACE = "zcs"

    def zcs(self, image, threshold=0.00001):
        if len(image.shape) == 2:
            kernel_name = "zcs1"
            global_size = image.size
        elif image.shape[2] == 2:
            kernel_name = "zcs2"
            global_size = image.size // 2
        elif image.shape[2] == 3:
            kernel_name = "zcs3"
            global_size = image.size // 3
        elif image.shape[2] == 4:
            kernel_name = "zcs4"
            global_size = image.size // 4

        handler = getattr(self.kmgr, kernel_name)
        return handler(threshold, image, global_size=global_size)

    def zcs_res(self, image, zcs, threshold=0.00001):
        if len(image.shape) == 2:
            kernel_name = "zcs1_res"
            global_size = image.size
        elif image.shape[2] == 2:
            kernel_name = "zcs2_res"
            global_size = image.size // 2
        elif image.shape[2] == 3:
            kernel_name = "zcs3_res"
            global_size = image.size // 3
        elif image.shape[2] == 4:
            kernel_name = "zcs4_res"
            global_size = image.size // 4

        handler = getattr(self.kmgr, kernel_name)
        handler(threshold, image, zcs, global_size=global_size)
