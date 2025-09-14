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

    def __init__(self, a, device_name: str | None = None, graphics: Graphics4 | None = None, chunk_size: int | None = 500):
        super().__init__(a, device_name=device_name, graphics=graphics, chunk_size=chunk_size)
        # Use 2D global size (width, height) for tiled graphics4 kernels
        if hasattr(self.a, "shape") and len(self.a.shape) >= 2:
            h, w = self.a.shape[0], self.a.shape[1]
            self.global_size = (w, h)
