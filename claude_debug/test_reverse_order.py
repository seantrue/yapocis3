#!/usr/bin/env python3
"""Test histogram in reverse order - 1500 first, then 256."""
import os
os.environ['YAPOCIS_BACKEND'] = 'cuda'

import numpy as np
from yapocis.lib.imageops import Imageops

print("Creating Imageops with FASTBINS=0...")
imageops = Imageops(FASTBINS=0)

a = np.random.uniform(0.1, 0.9, (2000, 2000)).astype(np.float32)

print("\n=== Call 1: histogram with 1500 bins ===")
h1 = imageops.histogram(a, n_bins=1500)
print(f"  h1.sum() = {h1.sum()}, expected = {a.shape[0]*a.shape[1]}")
print(f"  Match: {h1.sum() == a.shape[0]*a.shape[1]}")

print("\n=== Call 2: histogram with 256 bins ===")
h2 = imageops.histogram(a, n_bins=256)
print(f"  h2.sum() = {h2.sum()}, expected = {a.shape[0]*a.shape[1]}")
print(f"  Match: {h2.sum() == a.shape[0]*a.shape[1]}")

print("\n=== Comparison ===")
print(f"  Both match: {h1.sum() == a.shape[0]*a.shape[1] and h2.sum() == a.shape[0]*a.shape[1]}")
