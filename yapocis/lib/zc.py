"""
Low level wrapper for zero crossing detector kernel
"""
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

from .bound_kernels import BoundKernels


class ZC(BoundKernels):
    INTERFACE="zcs"

    def zcs(self, image, threshold=0.00001):
        if len(image.shape) == 2:
            handler = self.kmgr.zcs1
            global_size = image.size
        elif image.shape[2] == 2:
            handler = self.kmgr.zcs2
            global_size = image.size // 2
        elif image.shape[2] == 3:
            handler = self.kmgr.zcs3
            global_size = image.size // 3
        elif image.shape[2] == 4:
            handler = self.kmgr.zcs4
            global_size = image.size // 4
        return handler(threshold, image, global_size=global_size)


    def zcs_res(self, image, zcs, threshold=0.00001):
        if len(image.shape) == 2:
            handler = self.kmgr.zcs1_res
            global_size = image.size
        elif image.shape[2] == 2:
            handler = self.kmgr.zcs2_res
            global_size = image.size // 2
        elif image.shape[2] == 3:
            handler = self.kmgr.zcs3_res
            global_size = image.size // 3
        elif image.shape[2] == 4:
            handler = self.kmgr.zcs4_res
            global_size = image.size // 4
        handler(threshold, image, zcs, global_size=global_size)
