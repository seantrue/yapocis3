#!/usr/bin/env python3
"""Test global_size computation."""
import os
os.environ['YAPOCIS_BACKEND'] = 'cuda'

import numpy as np

# Patch to show global_size
from yapocis.cu import kernel_manager
original_grid_block = kernel_manager.CUDAProgram._grid_block

def debug_grid_block(self, gs, ls):
    result = original_grid_block(self, gs, ls)
    print(f"    _grid_block: gs={gs}, ls={ls} -> grid={result[0]}, block={result[1]}")
    return result

kernel_manager.CUDAProgram._grid_block = debug_grid_block

from yapocis.lib.imageops import Imageops

print("Creating Imageops...")
imageops = Imageops()

print("\nTest array: 10000 elements")
a = np.random.uniform(0.1, 0.9, 10000).astype(np.float32)

print("\n=== Call 1: 256 bins ===")
h1 = imageops.histogram(a, n_bins=256)
print(f"  Result sum: {h1.sum()}")

print("\n=== Call 2: 1500 bins ===")
h2 = imageops.histogram(a, n_bins=1500)
print(f"  Result sum: {h2.sum()}")

print("\n=== Call 3: 256 bins ===")
h3 = imageops.histogram(a, n_bins=256)
print(f"  Result sum: {h3.sum()}")
