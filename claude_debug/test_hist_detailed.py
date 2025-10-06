#!/usr/bin/env python3
"""Detailed histogram debugging."""
import os
os.environ['YAPOCIS_BACKEND'] = 'cuda'

import numpy as np
import cupy as cp

# Patch ensure_device to see what's happening
from yapocis.core import devbuf
original_ensure = devbuf.ensure_device

def debug_ensure(x):
    result = original_ensure(x)
    if hasattr(x, 'shape'):
        print(f"      ensure_device: {type(x).__name__} shape={x.shape} -> {type(result).__name__}")
        if isinstance(result, cp.ndarray):
            print(f"                     GPU sum={result.sum().get()}")
    return result

devbuf.ensure_device = debug_ensure

from yapocis.lib.imageops import Imageops

print("Creating Imageops...")
imageops = Imageops()

print("\nTest array: 10000 elements")
a = np.random.uniform(0.1, 0.9, 10000).astype(np.float32)

print("\n=== Call 1: 256 bins ===")
print("  Creating histogram buffer...")
h1 = imageops.histogram(a, n_bins=256)
print(f"  Result: sum={h1.sum()}")

print("\n=== Call 2: 512 bins ===")
print("  Creating histogram buffer...")
h2 = imageops.histogram(a, n_bins=512)
print(f"  Result: sum={h2.sum()}")

print("\n=== Call 3: 256 bins ===")
print("  Creating histogram buffer...")
h3 = imageops.histogram(a, n_bins=256)
print(f"  Result: sum={h3.sum()}")
