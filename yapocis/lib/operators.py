"""
Wrappers for binary operations of the form c = a + b, in both resident and inout form.
"""
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

import os
from yapocis.lib.bound_kernels import BoundKernels
from yapocis.utils.stereo_typing import *


class Operators(BoundKernels):
    INTERFACE = "operators"

    def __init__(self, *args, use_package: bool | None = None, **kwargs):
        # Operators context loaded automatically from operators.yml
        super(Operators, self).__init__(*args, **kwargs)

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
        from yapocis.kernelpack import get_kernelpack
        from yapocis.core.interface_context import load_interface_context

        # Load operator list from YAML
        context = load_interface_context("operators")
        operators = context.get('operators', [])

        # Get compiled package for current backend
        self.package = get_kernelpack('operators', self.kmgr)

        # Load all operator kernels
        kernel_names = [name for name, _ in operators]
        self.package.load(kernel_names)

        # Wrap package methods to match BoundKernels interface
        for name, _ in operators:
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
