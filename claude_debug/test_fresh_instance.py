#!/usr/bin/env python3
"""Test with fresh Imageops instance for each call."""
import os
os.environ['YAPOCIS_BACKEND'] = 'cuda'

import numpy as np
from yapocis.lib.imageops import Imageops

a = np.random.uniform(0.1, 0.9, (2000, 2000)).astype(np.float32)

print("=== Call 1: 256 bins with fresh instance ===")
imageops1 = Imageops(FASTBINS=0)
h1 = imageops1.histogram(a, n_bins=256)
print(f"  h1.sum() = {h1.sum()}, expected = {a.shape[0]*a.shape[1]}")

print("\n=== Call 2: 1500 bins with fresh instance ===")
imageops2 = Imageops(FASTBINS=0)
h2 = imageops2.histogram(a, n_bins=1500)
print(f"  h2.sum() = {h2.sum()}, expected = {a.shape[0]*a.shape[1]}")

print("\n=== Call 3: 256 bins with SAME instance as call 1 ===")
h3 = imageops1.histogram(a, n_bins=256)
print(f"  h3.sum() = {h3.sum()}, expected = {a.shape[0]*a.shape[1]}")

print("\n=== Call 4: 1500 bins with SAME instance as call 2 ===")
h4 = imageops2.histogram(a, n_bins=1500)
print(f"  h4.sum() = {h4.sum()}, expected = {a.shape[0]*a.shape[1]}")
