

#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import yapocis.rpc.programs
from yapocis.lib.yapocis_types import *

gradient_cl = None
program, kernels = yapocis.rpc.programs.import_program("gradient", gradient="gradient_cl")

def gradient(image:Array, reach:int=1) -> Tuple[Array,Array]:
    grad, theta = gradient_cl(image, reach)
    # TODO: gradientCL should not be returning nans, and is
    theta[np.where(np.isnan(theta))] = 0.0
    grad[np.where(np.isnan(grad))] = 0.0
    return grad, theta
