#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.

import time

import numpy
import numpy.linalg as la
import pyopencl as cl

# Make sure we allocate arrays of the type expected
a = numpy.random.rand(50000).astype(numpy.float32)
b = numpy.random.rand(50000).astype(numpy.float32)

# Talk to the compiler and linker, setup the runtime
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

t = time.time()

# Allocate buffers on the device to hold our data
# and copy it down to the device
mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, b.nbytes)

# Compile the kernel
prg = cl.Program(ctx, """
    __kernel void sum(__global const float *a,
    __global const float *b, __global float *c)
    {
      int gid = get_global_id(0);
      c[gid] = a[gid] + b[gid];
    }
    """).build()

# Run the kernel
prg.sum(queue, a.shape, None, a_buf, b_buf, dest_buf)

# And more biffer management to get the data back
a_plus_b = numpy.empty_like(a)
cl.enqueue_copy(queue, a_plus_b, dest_buf)

# But the answer comes back fast, and good.
print((la.norm(a_plus_b - (a+b)), la.norm(a_plus_b)))
print("Elapsed:", time.time() - t)