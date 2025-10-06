#!/usr/bin/env python3
"""Test atomicAdd with large histogram buffers."""
import numpy as np
import cupy as cp

# Test different histogram sizes
for n_bins in [256, 512, 1024, 1500, 2048]:
    print(f"\n=== Testing {n_bins} bins ===")

    # Create histogram buffer on GPU (initialized to zero)
    hst = cp.zeros(n_bins, dtype=cp.int32)
    print(f"  Buffer created: sum={hst.sum().get()}")

    # Simple kernel that does atomicAdd
    code = f"""
extern "C" __global__ void test_histogram(int *hst, int n_bins, int n_elements) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    float value = 0.5f;  // All threads write to same bucket
    int bucket = (int)((n_bins * value) / 1.0f);
    bucket = max(0, min(bucket, n_bins - 1));

    atomicAdd(&hst[bucket], 1);
}}
"""

    mod = cp.RawModule(code=code, options=('--std=c++14',))
    kern = mod.get_function('test_histogram')

    # Launch with 4M threads (same as test)
    n_elements = 4000000
    threads_per_block = 256
    n_blocks = (n_elements + threads_per_block - 1) // threads_per_block

    kern((n_blocks,), (threads_per_block,), (hst, n_bins, n_elements))
    cp.cuda.Stream.null.synchronize()

    result_sum = hst.sum().get()
    print(f"  After kernel: sum={result_sum} (expected: {n_elements})")
    if result_sum == n_elements:
        print(f"  ✓ PASS")
    else:
        print(f"  ✗ FAIL (got {result_sum}/{n_elements} = {100*result_sum/n_elements:.1f}%)")
