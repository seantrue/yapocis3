#!/usr/bin/env python3
"""Test histogram kernel."""
import os
os.environ['YAPOCIS_BACKEND'] = 'cuda'

import numpy as np
from yapocis.lib.imageops import Imageops

print("Creating Imageops...")
imageops = Imageops()

print("\nCreating small test array...")
a = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
print(f"Input: {a}")
print(f"Min: {a.min()}, Max: {a.max()}")

print("\nCalling histogram with n_bins=10...")
h = imageops.histogram(a, n_bins=10, low=0.0, high=1.0)
print(f"Histogram: {h}")
print(f"Sum: {h.sum()} (expected: {a.size})")

print("\n\nTesting with larger array...")
a2 = np.random.uniform(0.1, 0.9, 1000).astype(np.float32)
print(f"Array size: {a2.size}")
print(f"Range: [{a2.min():.3f}, {a2.max():.3f}]")

h2 = imageops.histogram(a2, n_bins=8, low=0.0, high=1.0)
print(f"Histogram: {h2}")
print(f"Sum: {h2.sum()} (expected: {a2.size})")
