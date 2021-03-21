"""
Wrappers for binary operations of the form c = a + b, in both resident and inout form.
"""
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from yapocis.lib.bound_kernels import BoundKernels
from yapocis.utils.typing import *

class Operators(BoundKernels):
    INTERFACE = "operators"
    OPERATORS = [("add", "+"), ("sub", "-"), ("mul", "*"), ("div", "/")]

    def __init__(self, *args, **kwargs):
        super(Operators, self).__init__(*args, operators=self.OPERATORS, **kwargs)
