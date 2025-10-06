#!/usr/bin/env python3
"""Manually call histogram kernel twice with different buffer sizes."""
import os
os.environ['YAPOCIS_BACKEND'] = 'cuda'
os.environ['YAPOCIS_DEBUG_KERNEL'] = '1'

import numpy as np
from yapocis.cu.kernel_manager import CUDAKernelManager

mgr = CUDAKernelManager()
prog = mgr.load_program("imageops", FASTBINS=0)

a = np.random.uniform(0.1, 0.9, (2000, 2000)).astype(np.float32)

print("=== Call 1: 256 bins ===")
hst1 = np.zeros(256, dtype=np.int32)
prog.histogram(a, 0.0, 1.0, hst1)
print(f"  Result: {hst1.sum()}\n")

print("=== Call 2: 1500 bins ===")
hst2 = np.zeros(1500, dtype=np.int32)
prog.histogram(a, 0.0, 1.0, hst2)
print(f"  Result: {hst2.sum()}\n")

print("=== Call 3: 256 bins again ===")
hst3 = np.zeros(256, dtype=np.int32)
prog.histogram(a, 0.0, 1.0, hst3)
print(f"  Result: {hst3.sum()}\n")
