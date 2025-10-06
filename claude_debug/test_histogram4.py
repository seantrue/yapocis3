#!/usr/bin/env python3
"""Test calling histogram twice."""
import os
os.environ['YAPOCIS_BACKEND'] = 'cuda'

import numpy as np
from yapocis.lib.imageops import Imageops

print("Creating Imageops with FASTBINS=0...")
imageops = Imageops(FASTBINS=0)

print("\nCreating test array...")
a = np.random.uniform(0.1, 0.9, (2000, 2000)).astype(np.float32)
print(f"Array shape: {a.shape}, size: {a.size}")

print("\n=== First call: 256 bins ===")
h1 = imageops.histogram(a, n_bins=256)
print(f"Sum: {h1.sum()}, non-zero: {np.count_nonzero(h1)}")

print("\n=== Second call: 1500 bins (same array) ===")
h2 = imageops.histogram(a, n_bins=1500)
print(f"Sum: {h2.sum()}, non-zero: {np.count_nonzero(h2)}")

print("\n=== Third call: 256 bins again ===")
h3 = imageops.histogram(a, n_bins=256)
print(f"Sum: {h3.sum()}, non-zero: {np.count_nonzero(h3)}")

print("\n=== Fourth call: 512 bins ===")
h4 = imageops.histogram(a, n_bins=512)
print(f"Sum: {h4.sum()}, non-zero: {np.count_nonzero(h4)}")
