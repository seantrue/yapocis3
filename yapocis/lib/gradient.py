"""
Wrapper for 2D gradient kernel
"""

#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from yapocis.utils.typing import *
from yapocis.rpc.kernel_manager import load_program
gradient_cl = None
gp = load_program("gradient")


def gradient(image:Image1, reach:int=1) -> Tuple[Image1,Image1]:
    """
    Return gradient and and angle array for an input 2d Array

    Args:
        image: 2D Array
        reach: to measure difference over

    Returns: tuple of 2D arrays

    """
    grad, theta = gp.gradient(image, reach)
    # TODO: gradientCL should not be returning nans, and is
    theta[np.where(np.isnan(theta))] = 0.0
    grad[np.where(np.isnan(grad))] = 0.0
    return grad, theta
