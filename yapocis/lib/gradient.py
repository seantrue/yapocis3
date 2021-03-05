"""
Wrapper for 2D gradient kernel
"""

#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from yapocis.utils.typing import *
from yapocis.rpc.kernel_manager import load_program

from .bound_kernels import BoundKernels

class Gradient(BoundKernels):
    def __init__(self, device_name: Optional[str]=None):
        super(Gradient, self).__init__("gradient", device_name=device_name)

    def gradient(self, image:Image1, reach:int=1) -> Tuple[Image1,Image1]:
        """
        Return gradient and and angle array for an input 2d Array

        Args:
            image: 2D Array
            reach: to measure difference over

        Returns: tuple of 2D arrays

        """
        grad, theta = self.kmgr.gradient(image, reach)
        theta[np.where(np.isnan(theta))] = 0.0
        grad[np.where(np.isnan(grad))] = 0.0
        return grad, theta
