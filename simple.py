import numpy as np
import pyopencl as cl

# Pick a device (prefer GPU), create context and queue
platforms = cl.get_platforms()
gpus = [d for p in platforms for d in p.get_devices(device_type=cl.device_type.GPU)]
devices = gpus or [d for p in platforms for d in p.get_devices()]
device = devices[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)
print(f"Using device: {device.name}")

# Create test data
a = np.random.random(1024).astype(np.float32)
b = np.random.random(1024).astype(np.float32)

# Create OpenCL buffers
mf = cl.mem_flags
buf_a = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=a)
buf_b = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=b)
buf_r = cl.Buffer(ctx, mf.WRITE_ONLY, size=a.nbytes)

# OpenCL kernel
kernel_src = r"""
__kernel void vector_add(__global const float* a,
                         __global const float* b,
                         __global float* result,
                         const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        result[gid] = a[gid] + b[gid];
    }
}
"""

# Compile and execute
prg = cl.Program(ctx, kernel_src).build()
kn = prg.vector_add

# Mirror Metal's (32,1,1) threadgroup size; pad global size if needed
local_size = 32
global_size = ((a.size + local_size - 1) // local_size) * local_size

kn.set_args(buf_a, buf_b, buf_r, np.int32(a.size))
cl.enqueue_nd_range_kernel(queue, kn, (global_size,), (local_size,))

# Get results
result = np.empty_like(a)
cl.enqueue_copy(queue, result, buf_r)
queue.finish()

print("Computation complete!")
# Optional sanity check:
print(np.allclose(result, a + b))
