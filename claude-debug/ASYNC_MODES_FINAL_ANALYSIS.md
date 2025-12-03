# Comprehensive Async Modes Analysis: All Combinations

## Executive Summary

Testing **all 4 combinations** of `YAPOCIS_ASYNC` and `YAPOCIS_ASYNC_WRITE` flags reveals that:

1. **The flags are independent** - they control different aspects of async operation
2. **Combining both flags** (`both` mode) provides the best performance for CUDA medium/large workloads
3. **Single-flag modes** can be optimal for specific scenarios

## Flag Definitions

| Flag | Controls | When Enabled |
|------|----------|--------------|
| `YAPOCIS_ASYNC` | Kernel execution | Kernels execute asynchronously without blocking |
| `YAPOCIS_ASYNC_WRITE` | Memory transfers | write() operations are asynchronous |

## Four Possible Modes

| Mode | YAPOCIS_ASYNC | YAPOCIS_ASYNC_WRITE | Description |
|------|---------------|---------------------|-------------|
| **sync** | 0 | 0 | All operations synchronous (baseline) |
| **async-exec** | 1 | 0 | Async kernel execution, sync memory |
| **async-write** | 0 | 1 | Sync kernel execution, async memory |
| **both** | 1 | 1 | Both kernels and memory are async |

## Performance Results

### Small Pipeline (3 ops × 512×512)

| Backend | Mode | Time | Speedup | Config |
|---------|------|------|---------|--------|
| **OCL** | sync | 8.31 ms | 1.00x | baseline |
| OCL | **async-exec** | 7.73 ms | **1.08x** ✓✓ | Best for OCL |
| OCL | async-write | 8.00 ms | 1.04x | |
| OCL | both | 7.87 ms | 1.06x | |
| **CUDA** | sync | 7.35 ms | 1.00x | baseline |
| CUDA | async-exec | 7.26 ms | 1.01x | |
| CUDA | **async-write** | 6.97 ms | **1.05x** ✓ | Best for CUDA |
| CUDA | both | 7.07 ms | 1.04x | |

### Medium Pipeline (6 ops × 512×512)

| Backend | Mode | Time | Speedup | Config |
|---------|------|------|---------|--------|
| **OCL** | sync | 18.24 ms | 1.00x | baseline |
| OCL | async-exec | 18.67 ms | 0.98x | |
| OCL | async-write | 19.05 ms | 0.96x | |
| OCL | both | 18.66 ms | 0.98x | ~sync |
| **CUDA** | sync | 12.17 ms | 1.00x | baseline |
| CUDA | async-exec | 10.59 ms | 1.15x ✓ | |
| CUDA | async-write | 9.71 ms | 1.25x ✓✓ | |
| CUDA | **both** | 9.68 ms | **1.26x** ✓✓✓ | **Best** |

### Large Image (3 ops × 1024×1024)

| Backend | Mode | Time | Speedup | Config |
|---------|------|------|---------|--------|
| **OCL** | sync | 24.88 ms | 1.00x | baseline |
| OCL | async-exec | 24.11 ms | 1.03x | |
| OCL | **async-write** | 23.68 ms | **1.05x** ✓ | Best for OCL |
| OCL | both | 25.06 ms | 0.99x | |
| **CUDA** | sync | 14.82 ms | 1.00x | baseline |
| CUDA | async-exec | 14.80 ms | 1.00x | |
| CUDA | async-write | 14.72 ms | 1.01x | |
| CUDA | **both** | 14.47 ms | **1.02x** ✓ | Best for CUDA |

## Key Findings

### 1. "Both" Mode is Best for CUDA Heavy Workloads

**CUDA Medium Pipeline**: 1.26x speedup with both flags enabled
- This is the **best overall result** across all tests
- Combining async execution + async writes provides maximum benefit

### 2. Single-Flag Modes Have Their Place

**OpenCL prefers single flags:**
- Small pipelines: async-exec only (1.08x)
- Large images: async-write only (1.05x)

**CUDA small workloads:**
- async-write only (1.05x) beats both (1.04x)

### 3. Workload Patterns

| Workload Type | Best Mode (OCL) | Best Mode (CUDA) |
|---------------|-----------------|------------------|
| Small pipeline (3 ops) | async-exec | async-write |
| Medium pipeline (6 ops) | sync (!) | **both (1.26x)** |
| Large image (1024²) | async-write | **both (1.02x)** |

### 4. When NOT to Use Async

**OpenCL medium pipeline**: All async modes are **slower** than sync
- Overhead from async coordination outweighs benefits
- Stick with sync for OCL moderate workloads

## Recommendations by Use Case

### For CUDA (Recommended for Production)

```python
# Small workloads (< 3 operations)
os.environ["YAPOCIS_ASYNC"] = "0"
os.environ["YAPOCIS_ASYNC_WRITE"] = "1"  # async-write only

# Medium to large workloads (6+ operations or large images)
os.environ["YAPOCIS_ASYNC"] = "1"
os.environ["YAPOCIS_ASYNC_WRITE"] = "1"  # both flags
```

### For OpenCL

```python
# Small pipelines (2-4 operations)
os.environ["YAPOCIS_ASYNC"] = "1"
os.environ.pop("YAPOCIS_ASYNC_WRITE", None)  # async-exec only

# Large images (≥1024×1024)
os.environ["YAPOCIS_ASYNC"] = "0"
os.environ["YAPOCIS_ASYNC_WRITE"] = "1"  # async-write only

# Medium workloads (5-10 operations)
os.environ["YAPOCIS_ASYNC"] = "0"
os.environ.pop("YAPOCIS_ASYNC_WRITE", None)  # sync - fastest!
```

## Understanding the Flags

### YAPOCIS_ASYNC (Kernel Execution)

**When enabled:**
- Kernels are enqueued without waiting for completion
- Multiple kernels can overlap execution
- Good for: Compute-bound operations, pipeline scenarios

**When disabled:**
- Each kernel completes before the next starts
- Simpler to debug and trace
- Good for: Single operations, debugging

### YAPOCIS_ASYNC_WRITE (Memory Transfers)

**When enabled:**
- write() returns immediately without waiting for GPU transfer
- Memory copies can overlap with kernel execution
- Good for: Memory-bound operations, large images

**When disabled:**
- write() waits for data to reach GPU before returning
- Ensures data is ready before kernel starts
- Good for: Small transfers where overhead > benefit

### Both Flags Together

**Maximum async mode:**
- Both kernel execution and memory transfers are async
- Best possible overlap of computation and data movement
- Highest performance potential but most complex to debug
- **CUDA shows excellent results** with this combination (up to 1.26x)

## Validation

All async modes produce **identical outputs** (max_diff=0.0) across all tests, confirming:
- Async is safe for production use
- No numerical accuracy loss
- Deterministic results

## Production Guidelines

### When to Use "Both" Mode (ASYNC=1, ASYNC_WRITE=1)

✓ **CUDA backend** with medium to heavy workloads
✓ Pipelines with **6+ operations**
✓ Processing **large images** (≥1024×1024)
✓ When **20%+ speedup** justifies complexity

### When to Use Single-Flag Modes

✓ **OpenCL backend** (better results with single flags)
✓ **Small workloads** (2-4 operations)
✓ When you need async but want **simpler debugging**

### When to Use Sync Mode

✓ **Single operations** on small images
✓ **Debugging** complex issues
✓ **OpenCL medium pipelines** (async is actually slower!)
✓ When **simplicity > 5% performance**

## The Surprising OCL Medium Pipeline Result

For OCL with 6 operations on 512×512:
- **Sync: 18.24 ms** (fastest!)
- async-exec: 18.67 ms (2.4% slower)
- async-write: 19.05 ms (4.4% slower)
- both: 18.66 ms (2.3% slower)

This suggests OCL driver overhead for async coordination exceeds the parallelism benefit at this workload size.

## Bottom Line

**For maximum performance:**

1. **CUDA + heavy workload**: Use `both` mode (ASYNC=1, ASYNC_WRITE=1) → **up to 1.26x**
2. **OCL + small pipeline**: Use `async-exec` only (ASYNC=1) → **up to 1.08x**
3. **OCL + large images**: Use `async-write` only (ASYNC_WRITE=1) → **up to 1.05x**
4. **OCL + medium pipeline**: Use `sync` → **fastest**

**The flags are independent** - you can mix and match based on your workload characteristics. Testing shows that combining both flags works best for CUDA, while single-flag modes often win for OpenCL.
