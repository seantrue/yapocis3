#!/usr/bin/env python3
"""Manual test of CUDA imageops kernel using CuPy directly."""
import numpy as np
import cupy as cp
from pathlib import Path

# Read the CUDA kernel source
kernel_path = Path('yapocis/cukernels/cuda_imageops.mako')
with open(kernel_path) as f:
    cuda_source = f.read()

print("=== CUDA Source (first 500 chars) ===")
print(cuda_source[:500])
print()

# Compile the module
print("Compiling CUDA module...")
mod = cp.RawModule(code=cuda_source, options=('--std=c++14',))
print("Compilation successful!")
print()

# Get the clamp_pixels kernel
print("Getting clamp_pixels function...")
clamp_kernel = mod.get_function('clamp_pixels')
print(f"Kernel: {clamp_kernel}")
print()

# Create test data
print("Creating test data...")
n = 1000
input_cpu = np.random.random(n).astype(np.float32)
output_cpu = np.zeros(n, dtype=np.float32)

print(f"Input - shape: {input_cpu.shape}, dtype: {input_cpu.dtype}")
print(f"Input range: [{input_cpu.min():.3f}, {input_cpu.max():.3f}]")
print()

# Transfer to GPU
print("Transferring to GPU...")
input_gpu = cp.asarray(input_cpu)
output_gpu = cp.asarray(output_cpu)
print(f"GPU input: {input_gpu.shape}, {input_gpu.dtype}")
print(f"GPU output: {output_gpu.shape}, {output_gpu.dtype}")
print()

# Set up kernel launch parameters
low = np.float32(0.1)
high = np.float32(0.9)
threads_per_block = 256
blocks = (n + threads_per_block - 1) // threads_per_block

print(f"Launch config: {blocks} blocks × {threads_per_block} threads")
print(f"Clamp range: [{low}, {high}]")
print()

# Call the kernel
print("Calling kernel...")
try:
    clamp_kernel(
        (blocks,),           # grid
        (threads_per_block,), # block
        (input_gpu, low, high, output_gpu)  # args
    )
    cp.cuda.Stream.null.synchronize()
    print("Kernel executed successfully!")
    print()

    # Get results
    result = output_gpu.get()
    print(f"Result range: [{result.min():.3f}, {result.max():.3f}]")
    print(f"Expected: all values in [0.1, 0.9]")

    # Verify
    if result.min() >= 0.1 - 1e-6 and result.max() <= 0.9 + 1e-6:
        print("✓ PASS: Values correctly clamped!")
    else:
        print("✗ FAIL: Values outside expected range")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
