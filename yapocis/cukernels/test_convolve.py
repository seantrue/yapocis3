import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

# First, render the Mako template
from mako.template import Template

conv = [0.1, 0.2, 0.4, 0.2, 0.1]
name = "convolution_1d"

template = Template(filename="cuda_convolution.mako")
cuda_code = template.render(conv=conv, name=name)

# Compile the CUDA code
mod = SourceModule(cuda_code)

# Get the kernel functions
convolution_1d = mod.get_function(name)
convolution_1d_fast = mod.get_function(name + "_fast")

# Prepare input data
width = 1024
h_a = np.random.rand(width).astype(np.float32)
h_ret = np.zeros_like(h_a)

# Allocate device memory
d_a = cuda.mem_alloc(h_a.nbytes)
d_ret = cuda.mem_alloc(h_ret.nbytes)

# Copy input data to device
cuda.memcpy_htod(d_a, h_a)

# set up grid and block dimensions
block_size = 256
grid_size = (width + block_size - 1) // block_size

# Launch the regular kernel
convolution_1d(np.int32(width), d_a, d_ret,
               block=(block_size, 1, 1), grid=(grid_size, 1))

# Copy results back to host
cuda.memcpy_dtoh(h_ret, d_ret)

print("Regular kernel result:", h_ret[:10])  # Print first 10 elements

# Reset h_ret
h_ret.fill(0)
cuda.memcpy_htod(d_ret, h_ret)

# Launch the fast kernel
chunk = 32
convolution_1d_fast(np.int32(width), d_a, d_ret, np.int32(chunk),
                    block=(block_size, 1, 1), grid=(grid_size // chunk, 1))

# Copy results back to host
cuda.memcpy_dtoh(h_ret, d_ret)

print("Fast kernel result:", h_ret[:10])  # Print first 10 elements

# Clean up
d_a.free()
d_ret.free()