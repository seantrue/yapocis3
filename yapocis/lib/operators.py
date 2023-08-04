"""
Wrappers for binary operations of the form c = a + b, in both resident and inout form.
"""
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

from yapocis.lib.bound_kernels import BoundKernels
from yapocis.utils.stereo_typing import *


class Operators(BoundKernels):
    INTERFACE = "operators"
    OPERATORS = [("add", "+"), ("sub", "-"), ("mul", "*"), ("div", "/")]

    def __init__(self, *args, **kwargs):
        super(Operators, self).__init__(*args, operators=self.OPERATORS, **kwargs)
