#!/usr/bin/env python3
"""Test histogram with 4M elements like the actual test."""
import os
os.environ['YAPOCIS_BACKEND'] = 'cuda'

import numpy as np
from yapocis.lib.imageops import Imageops

print("Creating Imageops with FASTBINS=0...")
imageops = Imageops(FASTBINS=0)

print("\nTest array: 4000000 elements (2000x2000)")
a = np.random.uniform(0.1, 0.9, (2000, 2000)).astype(np.float32)

print("\n=== Call 1: histogram with 256 bins ===")
h = imageops.histogram(a, n_bins=256)
print(f"  h.sum() = {h.sum()}, expected = {a.shape[0]*a.shape[1]}")
print(f"  Match: {h.sum() == a.shape[0]*a.shape[1]}")

print("\n=== Call 2: histogram with 1500 bins ===")
h1 = imageops.histogram(a, n_bins=1500)
print(f"  h1.sum() = {h1.sum()}, expected = {a.shape[0]*a.shape[1]}")
print(f"  Match: {h1.sum() == a.shape[0]*a.shape[1]}")

print("\n=== Comparison ===")
print(f"  h.sum() == h1.sum(): {h.sum() == h1.sum()}")
print(f"  h.sum() = {h.sum()}")
print(f"  h1.sum() = {h1.sum()}")
