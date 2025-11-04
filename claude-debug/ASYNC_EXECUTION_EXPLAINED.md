# Async Execution in Yapocis: GPU Frameworks and Synchronization

## Question 1: Under what GPU frameworks would using async help?

### Summary: OpenCL, Metal, and CUDA benefit; Numba does not

| Backend | Async Benefit? | Reason |
|---------|---------------|--------|
| **OpenCL** | ✅ **Yes (1.5-2x typical)** | Command queues with event-based synchronization |
| **Metal** | ✅ **Yes (1.5-2x typical)** | Command buffers can be enqueued without blocking |
| **CUDA** | ✅ **Yes (1.5-2x typical)** | Stream-based execution with async kernel launches |
| **Numba** | ❌ **No (API compatible only)** | CPU-based JIT compilation, inherently synchronous |

### Why GPU Backends Benefit from Async

GPU frameworks achieve parallelism through **asynchronous command queues**:

1. **CPU enqueues commands** → Fast (microseconds)
2. **GPU executes commands** → Slow (milliseconds)
3. **CPU continues working** → Overlap compute with command submission

**Traditional synchronous execution:**
```
CPU: [enqueue kernel 1] ← wait → [enqueue kernel 2] ← wait → [enqueue kernel 3] ← wait
GPU:                    [execute 1]                [execute 2]                [execute 3]
Timeline: ====W========|======W========|======W========|
          (W = CPU wasted time waiting)
```

**Async execution with batching:**
```
CPU: [enqueue 1][enqueue 2][enqueue 3] ← wait for all
GPU:            [execute 1][execute 2][execute 3]
Timeline: ===|=======================|
          (No CPU waiting, GPU keeps busy)
```

### Backend-Specific Implementation Details

#### OpenCL (PyOpenCL)
- **Mechanism:** `pyopencl.enqueue_*()` operations return `Event` objects
- **Async write:** `pyopencl.enqueue_copy(queue, buf, array)` without `.wait()`
- **Synchronization:** `pyopencl.wait_for_events(event_list)`
- **Example:**
```python
# Async path (from src/yapocis/ocl/device_manager.py:97-110)
def write_buffer(self, a: Array, wait: bool = True) -> pyopencl.Buffer:
    buf = self.ensure_buffer(a)
    if wait:
        pyopencl.enqueue_copy(self.queue, buf, a).wait()  # Sync
    else:
        pyopencl.enqueue_copy(self.queue, buf, a)  # Async - returns immediately
    return buf

def wait_events(self, events: list[pyopencl.Event]) -> None:
    if events:
        pyopencl.wait_for_events(events)  # Block until all complete
```

#### Metal (PyMetallic)
- **Mechanism:** Command buffers enqueued to command queue
- **Async write:** Buffer uploads enqueued without `waitUntilCompleted()`
- **Synchronization:** `command_buffer.waitUntilCompleted()`
- **Example:**
```python
# From src/yapocis/metallic/device_manager.py:123-135
def wait_events(self, events: list) -> None:
    if events:
        for event in events:
            if hasattr(event, 'waitUntilCompleted'):
                event.waitUntilCompleted()  # Block until GPU finishes
```

#### CUDA (PyCUDA)
- **Mechanism:** CUDA streams allow concurrent kernel execution
- **Async write:** Memory transfers to device without stream synchronization
- **Synchronization:** `stream.synchronize()`
- **Example:**
```python
# From src/yapocis/cu/device_manager.py:133-148
def wait_events(self, events: list) -> None:
    if events:
        unique_streams = set()
        for event in events:
            if hasattr(event, 'synchronize'):
                unique_streams.add(event)
        for stream in unique_streams:
            stream.synchronize()  # Block until stream completes
```

#### Numba (CPU)
- **No async benefit:** Numba executes on CPU threads/processes
- **Why it exists:** API compatibility - code written for GPU backends also runs on CPU
- **Use case:** Testing, development on machines without GPU

---

## Question 2: How does one defer running the function until the data is available?

### Answer: Async execution returns `None` immediately; synchronization happens explicitly

Yapocis provides two async execution patterns:

### Pattern 1: Manual Async with `async_exec=True`

```python
# From tests/test_async_execution.py:44-76
prog = core.load_program("imageops")
image = np.full((512, 512), 0.5, dtype=np.float32)

# Enqueue operations asynchronously - returns None immediately
result1 = prog.clamp_pixels(image, 0.0, 1.0, async_exec=True)  # Returns None
result2 = prog.clamp_pixels(image * 2, 0.0, 1.0, async_exec=True)  # Returns None
result3 = prog.clamp_pixels(image * 0.5, 0.0, 1.0, async_exec=True)  # Returns None

# Explicitly wait for ALL operations to complete
prog.wait_all()  # Blocks until GPU finishes all kernels

# Now read results from GPU
final_result = prog.read(output_buffer)
```

**Key points:**
- `async_exec=True` → kernel enqueued, returns `None` immediately
- GPU executes in background while CPU continues
- `wait_all()` → CPU blocks until GPU completes all pending operations
- `read()` → transfers result from GPU to CPU (only valid after `wait_all()`)

### Pattern 2: Context Manager with `async_mode()`

```python
# From tests/test_async_execution.py:78-101
prog = core.load_program("imageops")

with prog.async_mode():
    # All operations inside context are async
    result1 = prog.clamp_pixels(image, 0.0, 1.0)  # Returns None
    result2 = prog.clamp_pixels(image * 2, 0.0, 1.0)  # Returns None
    result3 = prog.clamp_pixels(image * 0.5, 0.0, 1.0)  # Returns None
# Context exit automatically calls wait_all()

# Results available here
final_result = prog.read(output_buffer)
```

**Key points:**
- Context manager sets `_default_async = True` for all operations
- Automatically calls `wait_all()` on context exit
- Cleaner API - no need to pass `async_exec=True` everywhere

### Implementation Details

From `src/yapocis/core/kmgr_base.py:264-290`:

```python
@contextmanager
def async_mode(self):
    """Context manager for async kernel execution."""
    old_default = self._default_async
    self._default_async = True  # Enable async for all kernels
    try:
        yield self
    finally:
        self.wait_all()  # Wait for all pending operations
        self._default_async = old_default  # Restore previous mode

def wait_all(self):
    """Wait for all pending async operations to complete."""
    if self.pending_events:
        if hasattr(self.dmgr, 'wait_events'):
            self.dmgr.wait_events(self.pending_events)  # Backend-specific wait
        self.pending_events.clear()
```

### Kernel Execution Flow

From `src/yapocis/core/kernel_base.py:280-303`:

```python
def __call__(self, *args, **kwargs):
    # Extract async flag (from parameter or context manager)
    async_exec = kwargs.pop("async_exec", self.kernel_manager._default_async)

    # ... prepare arguments, execute kernel ...

    # Handle async vs sync completion
    if async_exec:
        self._handle_async_completion()  # Track event, return None
        return None  # Can't return results yet - kernel still executing
    else:
        self._handle_sync_completion(kwargs)  # Wait for kernel, read results
        return self.prepare_return()  # Return actual data
```

### Real-World Example: Gaussian Blur Pipeline

From `src/yapocis/lib/gaussian.py:242-260`:

```python
def gauss_image(self, a: Image1, scale: int) -> Image1:
    g = self.get_gaussian_kernel(scale)
    aa = a.copy()
    b = np.empty_like(aa)
    g.write(aa)  # Upload to GPU
    g.write(b)   # Upload to GPU

    if self.use_async:
        # Async path: enqueue both convolutions without waiting
        with self.kmgr.async_mode():
            g.h(aa, b)  # Horizontal convolution (enqueued, doesn't wait)
            g.v(b, aa)  # Vertical convolution (enqueued, doesn't wait)
        # Context exit waits for both kernels to complete
    else:
        # Synchronous path (traditional)
        g.h(aa, b)  # Horizontal (waits for completion)
        g.v(b, aa)  # Vertical (waits for completion)

    g.read(aa)  # Download result from GPU
    return aa
```

**Performance benefit:**
- **Sync:** CPU waits after each kernel → GPU idle during buffer transfers
- **Async:** Both kernels enqueued back-to-back → GPU pipeline stays full

---

## When to Use Async Execution

### ✅ **Good Use Cases:**

1. **Multiple independent operations**
   ```python
   with prog.async_mode():
       blur1 = gaussian.gauss_image(img1, scale=3)  # GPU kernel 1
       blur2 = gaussian.gauss_image(img2, scale=3)  # GPU kernel 2
       blur3 = gaussian.gauss_image(img3, scale=3)  # GPU kernel 3
   # All three can execute concurrently on GPU
   ```

2. **Sequential pipeline with resident buffers**
   ```python
   with self.kmgr.async_mode():
       median.median3x3_res(input, output, resident=(input, output))
       median.median3x3_res(output, input, resident=(input, output))
       median.median3x3_res(input, output, resident=(input, output))
   # All enqueued without CPU waiting, GPU executes sequentially
   ```

3. **Iterative algorithms** (anisotropic diffusion, median filtering)
   ```python
   with self.kmgr.async_mode():
       for i in range(n):
           filter_res(input, scale, step, output, resident=(input, output))
           input, output = output, input
   # Entire loop enqueued, GPU executes all iterations
   ```

### ❌ **Not Beneficial:**

1. **Single kernel execution** - No parallelism opportunity
2. **Tiny kernels** - Command submission overhead exceeds async benefit
3. **CPU-bound preprocessing** - GPU waiting on CPU anyway
4. **Numba backend** - No true async support

---

## Performance Expectations

From `tests/test_async_execution.py:226-228`:

```
Speedup varies by backend:
  - GPU backends (Metal, OpenCL, CUDA): 1.5-2x typical
  - CPU backend (Numba): No speedup (API compatible only)
```

**Why 1.5-2x instead of linear scaling?**
- Command submission overhead still exists
- GPU must execute kernels sequentially (unless truly independent)
- Memory bandwidth can be bottleneck
- Benefit comes from **eliminating CPU wait time**, not parallelizing GPU work

---

## Summary

1. **Async benefits GPU frameworks** (OpenCL, Metal, CUDA) by eliminating CPU waiting
2. **Synchronization is explicit:** `wait_all()` or context manager exit
3. **Async operations return `None`** - results retrieved after synchronization
4. **Typical speedup:** 1.5-2x for GPU backends on multi-kernel pipelines
5. **Numba (CPU):** No benefit, but API-compatible for testing

The key insight: Async execution doesn't make individual kernels faster - it **eliminates idle CPU wait time** by batching command submissions, allowing the GPU to stay busy while the CPU prepares the next operation.
