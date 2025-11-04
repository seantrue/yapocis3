# Async Write Benchmark Analysis

## Problem Discovered

The OpenCL benchmark shows only **1.01x speedup** instead of the expected **1.5-2.0x speedup**.

## Root Cause

The benchmark is **not testing the right scenario** for async_write benefits.

### Current Benchmark Pattern (WRONG)

```python
for arr in arrays:
    result = prog.clamp_pixels(arr, 0.0, 1.0)
```

**What happens in each iteration:**

1. **Write buffer** (async if `async_write=True`)
   - `handle_in()` calls `self.write(arg, wait=not self.async_write)`
   - With `async_write=True`, this returns immediately
   - Buffer transfer happens in background

2. **Execute kernel**
   - Kernel is enqueued to GPU queue
   - Kernel starts executing

3. **SYNC POINT** (this is the problem!)
   - `_handle_sync_completion()` is called
   - Calls `self.evt.wait()` which blocks CPU
   - Waits for kernel to finish

4. **Return result**

**Result:** Each kernel call forces a synchronization point, which **negates any async buffer write benefit!**

### Why This Happens

From `src/yapocis/ocl/kernel.py:154-164`:

```python
def _handle_sync_completion(self, extra_kwargs):
    """Wait for OpenCL kernel completion."""
    poll = extra_kwargs.pop("poll", None)
    if poll:
        while True:
            status = self.evt.get_info(pyopencl.event_info.COMMAND_EXECUTION_STATUS)
            if status == pyopencl.command_execution_status.COMPLETE:
                break
            time.sleep(poll)
    else:
        self.evt.wait()  # ← BLOCKS HERE!
```

Every kernel call (without `async_exec=True`) blocks until completion.

### Timeline Visualization

**With async_write=False (sync writes):**
```
CPU:  [Write1][Wait] [Write2][Wait] [Write3][Wait] ...
GPU:  [Transfer1]     [Transfer2]     [Transfer3]     ...
      └─Kernel1       └─Kernel2       └─Kernel3
```

**With async_write=True BUT kernel sync (current benchmark):**
```
CPU:  [Write1][Execute1][Wait] [Write2][Execute2][Wait] ...
GPU:  [Transfer1─────────────] [Transfer2─────────────] ...
      └─Kernel1                └─Kernel2
```

The async write benefit is **completely hidden** by the kernel execution wait!

## When async_write Actually Helps

Async write provides benefit when you can **do other work while buffers transfer**.

### Scenario 1: Multiple Buffers Before Single Kernel

**Example:** Neural network layer with many weight buffers

```python
# Write all weight buffers (async)
for weight in weights:
    write(weight)  # Returns immediately

# Execute kernel that uses all weights
execute_layer()  # By this point, transfers have overlapped
```

**Timeline:**
```
CPU:  [Write1][Write2][Write3][Execute][Wait]
GPU:  [─Transfer1─────────────]
      [──Transfer2────────────]
      [───Transfer3───────────]
                              └─Kernel (all buffers ready)
```

### Scenario 2: Async Execution + Async Write

**Example:** Pipeline processing with async kernels

```python
for arr in arrays:
    write(arr)  # Async write
    execute_kernel(arr, async_exec=True)  # Async execution, no wait!

# Wait for all to complete
sync()
```

**Timeline:**
```
CPU:  [W1][E1][W2][E2][W3][E3]...[Sync]
GPU:  [─T1─────]
      └─K1      [─T2─────]
                └─K2      [─T3─────]
                          └─K3
```

Transfers and kernels can overlap!

### Scenario 3: Buffer Pre-staging

**Example:** Pre-load buffers for upcoming computation

```python
# Pre-write buffers for next frame (async)
for buf in next_frame_buffers:
    write(buf)  # Async

# Process current frame
process_current_frame()  # While next frame uploads

# By now, next frame buffers are ready!
```

## Why Gaussian Blur Doesn't Benefit

Looking at `Gaussian.gauss_image()` at lines 249-267:

```python
def gauss_image(self, a: Image1, scale: int) -> Image1:
    g = self.get_gaussian_kernel(scale)
    aa = a.copy()
    b = np.empty_like(aa)
    g.write(aa)  # Write buffer 1
    g.write(b)   # Write buffer 2

    if self.use_async:
        with self.kmgr.async_mode():
            g.h(aa, b)    # Horizontal convolution
            g.v(b, aa)    # Vertical convolution
    else:
        g.h(aa, b)        # ← With async_exec=False, this SYNCS!
        g.v(b, aa)        # ← This also SYNCS!

    g.read(aa)
    return aa
```

**With `async_write=True` but `async_exec=False`:**
- `g.write(aa)` returns immediately ✓
- `g.write(b)` returns immediately ✓
- `g.h(aa, b)` **blocks** waiting for kernel ✗
- No benefit realized!

**With `use_async=True` (async execution):**
- The `async_mode()` context manager enables async execution
- This is where you get the real pipeline benefit
- But this is testing `async_exec`, not `async_write`!

## Fix: Proper Benchmark

To properly test `async_write`, we need a scenario where buffer writes can overlap:

### Option 1: Multiple Writes Before Kernel

```python
# Measure time for multiple buffer uploads + one kernel execution
prog = core.load_program("imageops", async_write=True)

start = time.time()

# Write all buffers first (async)
buffers = []
for arr in arrays:
    # We need a way to write without executing!
    # Current API doesn't support this well...
    pass

# Execute one kernel that uses all buffers
result = prog.some_multi_buffer_kernel(buffers)

elapsed = time.time() - start
```

**Problem:** The yapocis API doesn't have multi-buffer kernels or separate write/execute phases!

### Option 2: Use async_exec with async_write

```python
prog = core.load_program("imageops", async_write=True)

start = time.time()

results = []
for arr in arrays:
    # Both write and execute are async
    result = prog.clamp_pixels(arr, 0.0, 1.0, async_exec=True)
    results.append(result)

# Sync once at the end
prog.sync()

elapsed = time.time() - start
```

This tests the **combined benefit** of async writes + async execution.

### Option 3: Pure Buffer Upload Benchmark

Test buffer upload speed in isolation:

```python
# Direct device manager test
from yapocis.core.manager import get_manager

mgr = get_manager("ocl")
device_mgr = mgr.device_mgr

arrays = [np.random.rand(512, 512).astype(np.float32) for _ in range(10)]

# Sync writes
start = time.time()
for arr in arrays:
    device_mgr.write_buffer(arr, wait=True)  # Sync
device_mgr.queue.finish()  # Ensure complete
sync_time = time.time() - start

# Async writes
start = time.time()
for arr in arrays:
    device_mgr.write_buffer(arr, wait=False)  # Async
device_mgr.queue.finish()  # Wait for all
async_time = time.time() - start

speedup = sync_time / async_time
```

This **directly tests** the write performance without kernel execution interference.

## Conclusion

The current benchmark is measuring the wrong thing. The synchronization points from kernel execution are hiding any async_write benefit.

**Recommendations:**

1. **Test pure buffer uploads** (Option 3) - This will show the true async_write benefit
2. **Test async_exec + async_write together** (Option 2) - This shows real-world pipeline benefit
3. **Update documentation** - Clarify that async_write alone provides minimal benefit; it shines when combined with async_exec or in specific multi-buffer scenarios

The **async_write implementation is correct** - the benchmark just isn't testing the right scenario!
