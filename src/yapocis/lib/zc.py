"""
Low level wrapper for zero crossing detector kernel
"""
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

import os
from .bound_kernels import BoundKernels


class ZC(BoundKernels):
    INTERFACE="zcs"

    def __init__(self, *args, use_package: bool | None = None, **kwargs):
        super().__init__(*args, **kwargs)

        # Determine whether to use compiled kernel package
        if use_package is None:
            use_package = os.environ.get('YAPOCIS_KERNELPACK', '').lower() in ('1', 'true', 'yes')

        # Attempt to use package if requested
        self.use_package = False
        if use_package:
            try:
                self._init_with_package()
                self.use_package = True
            except (ImportError, AttributeError):
                # Package not available, already using traditional path
                pass

    def _init_with_package(self):
        """Initialize using compiled kernel package (fast path)."""
        from yapocis.kernelpack import get_kernelpack, get_kernel_names

        # Get compiled package for current backend
        self.package = get_kernelpack('zcs', self.kmgr)
        kernel_names = get_kernel_names('zcs')
        self.package.load(kernel_names)

    def zcs(self, image, threshold=0.00001):
        if len(image.shape) == 2:
            kernel_name = 'zcs1'
            global_size = image.size
        elif image.shape[2] == 2:
            kernel_name = 'zcs2'
            global_size = image.size // 2
        elif image.shape[2] == 3:
            kernel_name = 'zcs3'
            global_size = image.size // 3
        elif image.shape[2] == 4:
            kernel_name = 'zcs4'
            global_size = image.size // 4

        if self.use_package:
            # Package uses generic method with kernel_name parameter
            return self.package.zcs(kernel_name, image, threshold)
        else:
            handler = getattr(self.kmgr, kernel_name)
            return handler(threshold, image, global_size=global_size)


    def zcs_res(self, image, zcs, threshold=0.00001):
        if len(image.shape) == 2:
            kernel_name = 'zcs1_res'
            global_size = image.size
        elif image.shape[2] == 2:
            kernel_name = 'zcs2_res'
            global_size = image.size // 2
        elif image.shape[2] == 3:
            kernel_name = 'zcs3_res'
            global_size = image.size // 3
        elif image.shape[2] == 4:
            kernel_name = 'zcs4_res'
            global_size = image.size // 4

        if self.use_package:
            # Package uses generic method with kernel_name parameter
            self.package.zcs_res(kernel_name, image, zcs, threshold)
        else:
            handler = getattr(self.kmgr, kernel_name)
            handler(threshold, image, zcs, global_size=global_size)
