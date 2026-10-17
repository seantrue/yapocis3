"""
Wrappers for binary operations of the form c = a + b, in both resident and inout form.
"""

#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.

from yapocis.lib.bound_kernels import BoundKernels


class Operators(BoundKernels):
    INTERFACE = "operators"
