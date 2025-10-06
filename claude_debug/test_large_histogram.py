#!/usr/bin/env python3
"""Test histogram with large array."""
import os
os.environ['YAPOCIS_BACKEND'] = 'cuda'

import numpy as np

# Patch to show global_size
from yapocis.cu import kernel_manager
original_grid_block = kernel_manager.CUDAProgram._grid_block

def debug_grid_block(self, gs, ls):
    result = original_grid_block(self, gs, ls)
    print(f"    _grid_block: gs={gs}, ls={ls} -> grid={result[0]}, block={result[1]}")
    grid_total = result[0][0] * result[0][1] * result[0][2] * result[1][0] * result[1][1] * result[1][2]
    print(f"    Total threads: {grid_total}")
    return result

kernel_manager.CUDAProgram._grid_block = debug_grid_block

from yapocis.lib.imageops import Imageops

print("Creating Imageops...")
imageops = Imageops()

print("\nTest array: 4000000 elements (2000x2000)")
a = np.random.uniform(0.1, 0.9, (2000, 2000)).astype(np.float32)
print(f"Array size: {a.size}")

print("\n=== Histogram with 256 bins ===")
h = imageops.histogram(a, n_bins=256)
print(f"  Result sum: {h.sum()} (expected: {a.size})")
print(f"  Non-zero bins: {np.count_nonzero(h)}")
