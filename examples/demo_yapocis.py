import yapocis.rpc.programs
from yapocis.rpc import engines, interfaces
import numpy
import numpy.linalg as la
import time

# We don't _need_ specify the dtype: that's done in the interface
a = numpy.random.rand(50000)
b = numpy.random.rand(50000)

t = time.time()
# We don't need to manage a compiler and linker, that can be done for us
demo = yapocis.rpc.programs.load_program("demo")

# We don't need to manage buffers, that can be done for us
a_plus_b = demo.sum(a,b)
print((la.norm(a_plus_b - (a+b)), la.norm(a_plus_b)))
print("Elapsed:", time.time() - t)

# And there are interesting stats available
print("Stats", demo)
