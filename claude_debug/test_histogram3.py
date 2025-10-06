#!/usr/bin/env python3
"""Debug histogram buffer initialization."""
import os
os.environ['YAPOCIS_BACKEND'] = 'cuda'

import numpy as np
import cupy as cp
from yapocis.cu.kernel_manager import CUDAKernelManager

print("Creating manager and loading program...")
mgr = CUDAKernelManager()
prog = mgr.load_program('imageops')

print("\nCreating test data...")
a = np.random.uniform(0.1, 0.9, 1000).astype(np.float32)
print(f"Input size: {a.size}")

# Test with 256 bins
print("\n=== Test 1: 256 bins ===")
hst1 = np.zeros(256, dtype=np.int32)
print(f"CPU histogram before: sum={hst1.sum()}, first 10: {hst1[:10]}")

# Convert to GPU
a_gpu = cp.asarray(a)
hst1_gpu = cp.asarray(hst1)
print(f"GPU histogram before: sum={hst1_gpu.sum()}, first 10: {hst1_gpu[:10].get()}")

# Call kernel directly
result = prog.histogram(a, 0.0, 1.0, hst1)
print(f"Result after call: {type(result)}")
print(f"CPU histogram after: sum={hst1.sum()}, non-zero: {np.count_nonzero(hst1)}")

# Test with 512 bins
print("\n=== Test 2: 512 bins ===")
hst2 = np.zeros(512, dtype=np.int32)
print(f"CPU histogram before: sum={hst2.sum()}, first 10: {hst2[:10]}")

# Convert to GPU
hst2_gpu = cp.asarray(hst2)
print(f"GPU histogram before: sum={hst2_gpu.sum()}, first 10: {hst2_gpu[:10].get()}")

# Call kernel directly
result = prog.histogram(a, 0.0, 1.0, hst2)
print(f"Result after call: {type(result)}")
print(f"CPU histogram after: sum={hst2.sum()}, non-zero: {np.count_nonzero(hst2)}")
