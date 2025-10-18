"""
Wrapper for 2D gradient kernel
"""

#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

import os
from yapocis.utils.stereo_typing import *

from .bound_kernels import BoundKernels


class Gradient(BoundKernels):
    INTERFACE = "gradient"

    def __init__(self, *args, use_package: bool | None = None, **kwargs):
        super(Gradient, self).__init__(*args, **kwargs)

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
        from yapocis.kernelpack import get_gradient_package

        # Get compiled package for current backend
        self.package = get_gradient_package(self.kmgr)
        self.package.load()

    def gradient(self, a, stride=3):
        if self.use_package:
            return self.package.gradient(a, stride)
        else:
            return self.kmgr.gradient(a, stride)
