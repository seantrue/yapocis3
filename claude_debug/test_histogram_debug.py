#!/usr/bin/env python3
"""Debug histogram calls with instrumentation."""
import os
os.environ['YAPOCIS_BACKEND'] = 'cuda'

import numpy as np
import cupy as cp

# Patch the kernel_manager to add debug output
from yapocis.cu import kernel_manager

original_call = None

def debug_wrapper(original_fn):
    def call_with_debug(*args, global_size=None, local_size=None, **kwargs):
        print(f"\n  [DEBUG] Kernel call:")
        print(f"    args: {len(args)}")
        for i, arg in enumerate(args):
            if hasattr(arg, 'shape'):
                print(f"      arg[{i}]: {type(arg).__name__} shape={arg.shape}, size={arg.size}")
            else:
                print(f"      arg[{i}]: {type(arg).__name__} value={arg}")
        print(f"    global_size (before): {global_size}")
        print(f"    local_size (before): {local_size}")

        result = original_fn(*args, global_size=global_size, local_size=local_size, **kwargs)

        print(f"    result type: {type(result)}")
        return result
    return call_with_debug

# Monkey patch
from yapocis.lib.imageops import Imageops

print("Creating Imageops...")
imageops = Imageops()

# Patch the histogram callable
original_hist = imageops.kmgr.callable['histogram']
imageops.kmgr.callable['histogram'] = debug_wrapper(original_hist)

print("\nCreating test array...")
a = np.random.uniform(0.1, 0.9, (100, 100)).astype(np.float32)
print(f"Array shape: {a.shape}, size: {a.size}")

print("\n=== Call 1: 256 bins ===")
h1 = imageops.histogram(a, n_bins=256)
print(f"Result sum: {h1.sum()}")

print("\n=== Call 2: 512 bins ===")
h2 = imageops.histogram(a, n_bins=512)
print(f"Result sum: {h2.sum()}")

print("\n=== Call 3: 256 bins again ===")
h3 = imageops.histogram(a, n_bins=256)
print(f"Result sum: {h3.sum()}")
