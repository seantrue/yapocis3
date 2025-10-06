#!/usr/bin/env python3
"""Test if histogram buffers are properly initialized on GPU."""
import numpy as np
import cupy as cp

print("Testing buffer initialization...")

for n_bins in [256, 512, 1024, 1500]:
    print(f"\n=== {n_bins} bins ===")

    # Create on CPU
    hst_cpu = np.zeros(n_bins, dtype=np.int32)
    print(f"  CPU: sum={hst_cpu.sum()}, first 5: {hst_cpu[:5]}")

    # Transfer to GPU via asarray
    hst_gpu = cp.asarray(hst_cpu)
    print(f"  GPU (via asarray): sum={hst_gpu.sum().get()}, first 5: {hst_gpu[:5].get()}")

    # Create directly on GPU
    hst_gpu2 = cp.zeros(n_bins, dtype=cp.int32)
    print(f"  GPU (via zeros): sum={hst_gpu2.sum().get()}, first 5: {hst_gpu2[:5].get()}")

    # After kernel, check if we can read it back
    # Simulate some atomicAdds
    code = """
    extern "C" __global__ void fill_hist(int *hst, int n_bins) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < 10000) {  // 10k threads
            int bucket = i % n_bins;
            atomicAdd(&hst[bucket], 1);
        }
    }
    """
    mod = cp.RawModule(code=code, options=('--std=c++14',))
    kern = mod.get_function('fill_hist')

    # Reset buffer
    hst_gpu.fill(0)

    # Call kernel
    kern((40,), (256,), (hst_gpu, n_bins))
    cp.cuda.Stream.null.synchronize()

    result_sum = hst_gpu.sum().get()
    expected = 10000
    print(f"  After kernel: sum={result_sum} (expected: {expected}) - {'✓ PASS' if result_sum == expected else '✗ FAIL'}")
