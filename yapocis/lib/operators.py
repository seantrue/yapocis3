"""
Wrappers for binary operations of the form c = a + b, in both resident and inout form.
"""
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

import os
from yapocis.lib.bound_kernels import BoundKernels
from yapocis.utils.stereo_typing import *


class Operators(BoundKernels):
    INTERFACE = "operators"
    OPERATORS = [("add", "+"), ("sub", "-"), ("mul", "*"), ("divop", "/")]

    def __init__(self, *args, use_package: bool | None = None, **kwargs):
        super(Operators, self).__init__(*args, operators=self.OPERATORS, **kwargs)

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
        from yapocis.kernelpack import get_operators_package

        # Get compiled package for current backend
        self.package = get_operators_package(self.kmgr)

        # Load all operator kernels
        kernel_names = [name for name, _ in self.OPERATORS]
        self.package.load(kernel_names)

        # Wrap package methods to match BoundKernels interface
        for name, _ in self.OPERATORS:
            # Create wrapper for main operator
            setattr(self, name, self._make_operator_wrapper(name))
            # Create wrapper for _res variant
            setattr(self, f"{name}_res", self._make_operator_res_wrapper(name))

        # Also create div alias for divop
        self.div = self._make_operator_wrapper('div')
        self.div_res = self._make_operator_res_wrapper('div')

    def _make_operator_wrapper(self, kernel_name: str):
        """Create wrapper for operator() method."""
        def wrapper(input1: Array, input2: Array) -> Array:
            return self.package.operator(kernel_name, input1, input2)
        return wrapper

    def _make_operator_res_wrapper(self, kernel_name: str):
        """Create wrapper for operator_res() method."""
        def wrapper(input1: Array, input2: Array, output: Array) -> None:
            return self.package.operator_res(kernel_name, input1, input2, output)
        return wrapper
