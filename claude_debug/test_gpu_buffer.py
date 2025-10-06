#!/usr/bin/env python3
"""Test GPU buffer initialization."""
import numpy as np
import cupy as cp

print("Test 1: Create 256-element buffer")
h1 = np.zeros(256, dtype=np.int32)
h1_gpu = cp.asarray(h1)
print(f"  CPU: sum={h1.sum()}")
print(f"  GPU: sum={h1_gpu.sum().get()}")

print("\nTest 2: Create 512-element buffer")
h2 = np.zeros(512, dtype=np.int32)
h2_gpu = cp.asarray(h2)
print(f"  CPU: sum={h2.sum()}")
print(f"  GPU: sum={h2_gpu.sum().get()}")

print("\nTest 3: Create 1500-element buffer")
h3 = np.zeros(1500, dtype=np.int32)
h3_gpu = cp.asarray(h3)
print(f"  CPU: sum={h3.sum()}")
print(f"  GPU: sum={h3_gpu.sum().get()}")

print("\nTest 4: atomicAdd simulation")
h4 = cp.zeros(1500, dtype=cp.int32)
print(f"  Before: sum={h4.sum().get()}")

# Simple kernel to test atomicAdd
code = """
extern "C" __global__ void test_atomic(int *hst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&hst[i % 1500], 1);
    }
}
"""
mod = cp.RawModule(code=code, options=('--std=c++14',))
kern = mod.get_function('test_atomic')

# Launch with 10000 threads
kern((40,), (256,), (h4, 10000))
cp.cuda.Stream.null.synchronize()

print(f"  After: sum={h4.sum().get()} (expected: 10000)")
print(f"  Non-zero: {cp.count_nonzero(h4).get()}")
