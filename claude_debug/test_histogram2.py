#!/usr/bin/env python3
"""Test histogram with different bin sizes."""
import os
os.environ['YAPOCIS_BACKEND'] = 'cuda'

import numpy as np
from yapocis.lib.imageops import Imageops

print("Creating Imageops...")
imageops = Imageops(FASTBINS=0)

print("\nCreating test array...")
a = np.random.uniform(0.1, 0.9, (2000, 2000)).astype(np.float32)
print(f"Array shape: {a.shape}, size: {a.size}")

print("\nTest 1: histogram with 256 bins...")
h1 = imageops.histogram(a, n_bins=256)
print(f"Sum: {h1.sum()} (expected: {a.size})")
print(f"Non-zero bins: {np.count_nonzero(h1)}")

print("\nTest 2: histogram with 1500 bins...")
h2 = imageops.histogram(a, n_bins=1500)
print(f"Sum: {h2.sum()} (expected: {a.size})")
print(f"Non-zero bins: {np.count_nonzero(h2)}")

print("\nTest 3: histogram with 512 bins...")
h3 = imageops.histogram(a, n_bins=512)
print(f"Sum: {h3.sum()} (expected: {a.size})")
print(f"Non-zero bins: {np.count_nonzero(h3)}")

print("\nTest 4: histogram with 1024 bins...")
h4 = imageops.histogram(a, n_bins=1024)
print(f"Sum: {h4.sum()} (expected: {a.size})")
print(f"Non-zero bins: {np.count_nonzero(h4)}")

print("\nTest 5: histogram with 2048 bins...")
h5 = imageops.histogram(a, n_bins=2048)
print(f"Sum: {h5.sum()} (expected: {a.size})")
print(f"Non-zero bins: {np.count_nonzero(h5)}")
