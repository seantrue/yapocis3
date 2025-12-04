"""
Wrappers for binary operations of the form c = a + b, in both resident and inout form.
"""

#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

from yapocis.lib.bound_kernels import BoundKernels


class Operators(BoundKernels):
    INTERFACE = "operators"
