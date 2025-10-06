#!/usr/bin/env python3
"""Log actual kernel arguments."""
import os
os.environ['YAPOCIS_BACKEND'] = 'cuda'

import numpy as np
import cupy as cp

# Patch RawKernel to see what's being passed
original_call = cp._core.raw.RawKernel.__call__

def debug_call(self, grid, block, args, **kwargs):
    print(f"  CUDA kernel call:")
    print(f"    grid: {grid}")
    print(f"    block: {block}")
    print(f"    args ({len(args)}):")
    for i, arg in enumerate(args):
        if isinstance(arg, cp.ndarray):
            print(f"      [{i}] cupy array: shape={arg.shape}, dtype={arg.dtype}, size={arg.size}")
        elif isinstance(arg, np.ndarray):
            print(f"      [{i}] numpy array: shape={arg.shape}, dtype={arg.dtype}, size={arg.size}")
        elif isinstance(arg, (np.int32, np.float32, np.int64, np.float64)):
            print(f"      [{i}] {type(arg).__name__}: {arg}")
        else:
            print(f"      [{i}] {type(arg)}: {arg}")
    return original_call(self, grid, block, args, **kwargs)

cp._core.raw.RawKernel.__call__ = debug_call

from yapocis.lib.imageops import Imageops

print("Creating Imageops...")
imageops = Imageops()

a = np.random.uniform(0.1, 0.9, (2000, 2000)).astype(np.float32)

print("\n=== Call 1: 256 bins ===")
h1 = imageops.histogram(a, n_bins=256)
print(f"Result: {h1.sum()}\n")

print("\n=== Call 2: 1500 bins ===")
h2 = imageops.histogram(a, n_bins=1500)
print(f"Result: {h2.sum()}\n")
