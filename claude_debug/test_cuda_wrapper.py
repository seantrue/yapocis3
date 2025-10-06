#!/usr/bin/env python3
"""Test CUDA kernel through yapocis wrapper."""
import os
os.environ['YAPOCIS_BACKEND'] = 'cuda'

import numpy as np
import cupy as cp
from yapocis.cu.kernel_manager import CUDAKernelManager
from yapocis.core.interface import load_interface, get_interface

print("Creating CUDAKernelManager...")
mgr = CUDAKernelManager()
print(f"Templates dir: {mgr.templates_dir}")
print()

print("Loading imageops program...")
prog = mgr.load_program('imageops')
print(f"Program: {prog}")
print(f"Program name: {prog.name}")
print(f"Callable keys: {list(prog.callable.keys())}")
print()

print("Creating test data...")
a = np.random.random(100).astype(np.float32)
print(f"Input shape: {a.shape}, dtype: {a.dtype}, range: [{a.min():.3f}, {a.max():.3f}]")
print()

print("Calling prog.clamp_pixels via __getattr__...")
kernel_fn = prog.clamp_pixels
print(f"Got kernel function: {kernel_fn}")
print()

print("Examining what the wrapper will do...")
print("Arguments to pass:")
print(f"  a: {type(a)}, shape={a.shape}, dtype={a.dtype}")
print(f"  low: 0.1 (float)")
print(f"  high: 0.9 (float)")
print()

print("Calling kernel function with args...")
try:
    result = kernel_fn(a, 0.1, 0.9)
    print(f"SUCCESS! Result: {result}")
    if result is not None:
        print(f"Result type: {type(result)}")
        if hasattr(result, 'shape'):
            print(f"Result shape: {result.shape}")
            print(f"Result dtype: {result.dtype}")
            # Get back to CPU
            if isinstance(result, cp.ndarray):
                result_cpu = result.get()
            else:
                result_cpu = result
            print(f"Result range: [{result_cpu.min():.3f}, {result_cpu.max():.3f}]")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
