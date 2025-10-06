#!/usr/bin/env python3
"""Test histogram with ONLY 1500 bins (no 256 call first)."""
import os
os.environ['YAPOCIS_BACKEND'] = 'cuda'
os.environ['YAPOCIS_DEBUG_KERNEL'] = '1'

import numpy as np
from yapocis.lib.imageops import Imageops

print("Creating Imageops with FASTBINS=0...")
imageops = Imageops(FASTBINS=0)

print("\nTest array: 4000000 elements (2000x2000)")
a = np.random.uniform(0.1, 0.9, (2000, 2000)).astype(np.float32)

print("\n=== Call: histogram with 1500 bins (ONLY) ===")
h = imageops.histogram(a, n_bins=1500)
print(f"  h.sum() = {h.sum()}, expected = {a.shape[0]*a.shape[1]}")
print(f"  Match: {h.sum() == a.shape[0]*a.shape[1]}")
print(f"  First 10 bins: {h[:10]}")
print(f"  Last 10 bins: {h[-10:]}")
