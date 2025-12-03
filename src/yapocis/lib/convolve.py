#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from yapocis.lib.bound_kernels import BoundKernels
from yapocis.utils.stereo_typing import *

class Convolve1D(BoundKernels):
    INTERFACE = "convolve"

    def __init__(self, *args, name=None, conv=None, **kwargs):
        assert name is not None and conv is not None
        super(Convolve1D, self).__init__(*args, name=name, conv=conv, **kwargs)



