#!/usr/bin/env python3
"""
Test async execution support in yapocis kernel managers.

Demonstrates:
1. Synchronous execution (baseline)
2. Manual async execution with async_exec=True
3. Context manager async_mode() for automatic async
4. Performance comparison showing speedup
"""

import time
import numpy as np
from yapocis import core

def test_sync_execution():
    """Baseline: synchronous execution (default behavior)."""
    print("\n" + "="*70)
    print("TEST 1: Synchronous Execution (Baseline)")
    print("="*70)

    prog = core.load_program("imageops")

    # Create test data
    image = np.random.rand(512, 512).astype(np.float32)
    output1 = np.zeros_like(image)
    output2 = np.zeros_like(image)
    output3 = np.zeros_like(image)

    # Execute synchronously (current behavior)
    start = time.time()
    output1 = prog.clamp_pixels(image, 0.0, 1.0)  # Blocks, returns result
    output2 = prog.stretch(output1, 0.0, 1.0, 0.0, 2.0)  # Blocks, returns result
    output3 = prog.clamp_pixels(output2, 0.0, 1.0)  # Blocks, returns result
    elapsed_sync = time.time() - start

    print(f"Synchronous execution: {elapsed_sync*1000:.2f}ms")
    print(f"  - 3 kernels executed sequentially")
    print(f"  - CPU blocked after each kernel")
    print(f"  - Total: ~{elapsed_sync*1000:.2f}ms")

    return elapsed_sync

def test_manual_async():
    """Manual async using async_exec=True parameter."""
    print("\n" + "="*70)
    print("TEST 2: Manual Async Execution (async_exec=True)")
    print("="*70)

    prog = core.load_program("imageops")

    # Create test data
    image = np.random.rand(512, 512).astype(np.float32)
    output1 = np.zeros_like(image)
    output2 = np.zeros_like(image)
    output3 = np.zeros_like(image)

    # Execute asynchronously with manual flag
    start = time.time()
    # Note: async operations return None, so we track buffers manually
    prog.clamp_pixels(image, 0.0, 1.0, async_exec=True)  # Returns None (async)
    prog.stretch(output1, 0.0, 1.0, 0.0, 2.0, async_exec=True)  # Returns None (async)
    prog.clamp_pixels(output2, 0.0, 1.0, async_exec=True)  # Returns None (async)

    # Wait for all operations to complete
    prog.wait_all()
    elapsed_async = time.time() - start

    print(f"Manual async execution: {elapsed_async*1000:.2f}ms")
    print(f"  - 3 kernels enqueued without waiting")
    print(f"  - Single wait_all() at end")
    print(f"  - Total: ~{elapsed_async*1000:.2f}ms")

    return elapsed_async

def test_context_manager_async():
    """Context manager async using async_mode()."""
    print("\n" + "="*70)
    print("TEST 3: Context Manager Async (async_mode())")
    print("="*70)

    prog = core.load_program("imageops")

    # Create test data
    image = np.random.rand(512, 512).astype(np.float32)
    output1 = np.zeros_like(image)
    output2 = np.zeros_like(image)
    output3 = np.zeros_like(image)

    # Execute asynchronously with context manager
    start = time.time()
    with prog.async_mode():
        prog.clamp_pixels(image, 0.0, 1.0)  # Async (no explicit flag needed)
        prog.stretch(output1, 0.0, 1.0, 0.0, 2.0)  # Async
        prog.clamp_pixels(output2, 0.0, 1.0)  # Async
    # Context exit calls wait_all() automatically
    elapsed_ctx = time.time() - start

    print(f"Context manager async: {elapsed_ctx*1000:.2f}ms")
    print(f"  - Pythonic API with context manager")
    print(f"  - Automatic wait on context exit")
    print(f"  - Total: ~{elapsed_ctx*1000:.2f}ms")

    return elapsed_ctx

def test_pipeline_async():
    """Demonstrate pipelined async execution with multiple operations."""
    print("\n" + "="*70)
    print("TEST 4: Pipeline with Multiple Operations")
    print("="*70)

    prog = core.load_program("imageops")

    # Create larger dataset
    image = np.random.rand(1024, 1024).astype(np.float32)
    outputs = [np.zeros_like(image) for _ in range(10)]

    # Synchronous baseline
    start = time.time()
    temp = image.copy()
    for i in range(10):
        temp = prog.clamp_pixels(temp, 0.0, 1.0)  # Each call blocks
    elapsed_sync = time.time() - start

    # Async execution
    start = time.time()
    with prog.async_mode():
        temp = image
        for i in range(10):
            prog.clamp_pixels(temp, 0.0, 1.0)  # Enqueue without blocking
    elapsed_async = time.time() - start

    print(f"Synchronous (10 ops): {elapsed_sync*1000:.2f}ms")
    print(f"Async (10 ops):       {elapsed_async*1000:.2f}ms")
    print(f"Speedup: {elapsed_sync/elapsed_async:.2f}x")

    return elapsed_sync, elapsed_async

def main():
    print("\n" + "="*70)
    print("Yapocis Async Execution Test Suite")
    print("="*70)

    try:
        # Run all tests
        sync_time = test_sync_execution()
        manual_time = test_manual_async()
        context_time = test_context_manager_async()
        pipeline_sync, pipeline_async = test_pipeline_async()

        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"\nSingle operations (3 kernels):")
        print(f"  Synchronous:       {sync_time*1000:.2f}ms (baseline)")
        print(f"  Manual async:      {manual_time*1000:.2f}ms ({sync_time/manual_time:.2f}x speedup)")
        print(f"  Context manager:   {context_time*1000:.2f}ms ({sync_time/context_time:.2f}x speedup)")

        print(f"\nPipeline (10 operations):")
        print(f"  Synchronous:       {pipeline_sync*1000:.2f}ms (baseline)")
        print(f"  Async:             {pipeline_async*1000:.2f}ms ({pipeline_sync/pipeline_async:.2f}x speedup)")

        print(f"\nExpected benefits:")
        print(f"  - Reduced CPU/GPU synchronization overhead")
        print(f"  - GPU stays busy instead of idling between operations")
        print(f"  - Better throughput for pipeline workloads")
        print(f"  - Clean Pythonic API with context managers")

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
