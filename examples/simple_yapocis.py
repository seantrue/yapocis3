#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.

import time

import numpy as np
import numpy.linalg as la

from yapocis.ocl.kernel_manager import load_program

# We don't _need_ specify the dtype: that's done in the interface
a = np.random.rand(50000)
b = np.random.rand(50000).astype(np.float32)

t = time.time()
# We don't need to manage a compiler and linker, that can be done for us
demo = load_program("demo")

# We don't need to manage buffers, that can be done for us
a_plus_b = demo.sum(a,b)
print((la.norm(a_plus_b - (a+b)), la.norm(a_plus_b)))
print("Elapsed:", time.time() - t)

# And there are interesting stats available
print("Stats", demo)
