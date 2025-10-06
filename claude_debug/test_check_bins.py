#!/usr/bin/env python3
"""Check which bins are non-zero."""
import os
os.environ['YAPOCIS_BACKEND'] = 'cuda'

import numpy as np
from yapocis.lib.imageops import Imageops

imageops = Imageops(FASTBINS=0)
a = np.random.uniform(0.1, 0.9, (2000, 2000)).astype(np.float32)

print("=== 1500 bins ===")
h = imageops.histogram(a, n_bins=1500)
print(f"h.sum() = {h.sum()}")
print(f"h.dtype = {h.dtype}")
print(f"h.shape = {h.shape}")
print(f"Non-zero bins: {np.count_nonzero(h)}")
print(f"Min value: {h.min()}, Max value: {h.max()}")

# Find indices of non-zero bins
nonzero_idx = np.nonzero(h)[0]
if len(nonzero_idx) > 0:
    print(f"First 10 non-zero indices: {nonzero_idx[:10]}")
    print(f"Values at those indices: {h[nonzero_idx[:10]]}")
else:
    print("NO NON-ZERO BINS!")
    
# Check the histogram object itself
print(f"\nHistogram type: {type(h)}")
print(f"Is it a cupy array? {hasattr(h, 'device')}")
