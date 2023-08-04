"""
High level wrappers around the OpenCL graphics kernels.
"""
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.
import numpy as np

from yapocis.lib.abstract_graphics import AbstractKernelDraw
from yapocis.lib.graphics4 import Graphics4


class KernelDraw4(AbstractKernelDraw):
    COLOR_WIDTH = 4
    GRAPHICS = Graphics4

