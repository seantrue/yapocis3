"""
High level wrappers around the OpenCL graphics kernels.
"""
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.
from yapocis.lib.abstract_graphics import AbstractKernelDraw
from yapocis.lib.graphics import Graphics1

class KernelDraw1(AbstractKernelDraw):
    COLOR_WIDTH = 1
    GRAPHICS = Graphics1
