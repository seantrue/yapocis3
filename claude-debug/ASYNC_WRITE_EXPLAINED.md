# Async Write in Yapocis: Buffer Transfer Synchronization

## Question: Under what GPU frameworks does `async_write` help, and how does it defer execution?

## Summary

| Backend | async_write Benefit? | Mechanism |
|---------|---------------------|-----------|
| **OpenCL** | ✅ **Yes** | Command queue ordering - kernel waits automatically |
| **CUDA** | ✅ **Yes** | Stream ordering - kernel waits automatically |
| **Metal** | ❌ **No** | `wait` parameter ignored, transfers always synchronous |
| **Numba** | ❌ **No** | CPU execution, no GPU transfers |

## The Key Insight: Command Queue Ordering

GPU frameworks use **command queues** (OpenCL) or **streams** (CUDA) that guarantee **execution order**:

```
Command Queue: [Transfer A→GPU] → [Kernel Execute] → [Transfer GPU→B]
               \_______________ GPU ensures proper ordering ______________/
```

When you enqueue operations to the same queue/stream:
1. **Transfer completes first** (GPU hardware enforces this)
2. **Kernel waits for transfer** (automatic, no CPU involvement)
3. **CPU can continue** (if async - doesn't wait for GPU)

### Visual Comparison

**Synchronous write (`async_write=False`)**:
```
CPU: [write buffer]──WAIT──┐    [call kernel]──WAIT──┐
                            │                         │
GPU:                   [transfer]              [execute kernel]

Timeline: CPU is BLOCKED twice (wasted time)
```

**Async write (`async_write=True`)**:
```
CPU: [enqueue write][enqueue kernel]──WAIT for both──┐
                                                      │
GPU:            [transfer]──[execute kernel]         │
                            ^automatic wait           │
                                                      │
Timeline: CPU only blocks ONCE at the end
```

## Implementation Details

### OpenCL: Command Queue Ordering

From `src/yapocis/ocl/device_manager.py:97-110`:

```python
def write_buffer(self, a: Array, wait: bool = True) -> pyopencl.Buffer:
    """Write numpy array to PyOpenCL buffer."""
    buf = self.ensure_buffer(a)
    # ... reshape for flat copy ...

    if wait:
        # Synchronous: CPU blocks until transfer completes
        pyopencl.enqueue_copy(self.queue, buf, a).wait()
    else:
        # Async: Transfer enqueued, CPU returns immediately
        # The kernel will automatically wait for this transfer
        pyopencl.enqueue_copy(self.queue, buf, a)  # Returns Event

    return buf
```

**How deferral works:**
1. `async_write=True` → `write(wait=False)` → `enqueue_copy()` without `.wait()`
2. Kernel call → `kernel(queue, ...)` enqueued to **same queue**
3. OpenCL command queue **guarantees ordering**:
   - Transfer executes first
   - Kernel waits automatically (GPU-side)
   - CPU doesn't wait (continues immediately)

From `src/yapocis/ocl/kernel.py:135-140`:
```python
def _execute_kernel_with_args(self, prepared_args, original_args, extra_kwargs):
    """Execute OpenCL kernel with prepared arguments."""
    self.evt = self.kernel(
        self.kernel_manager.dmgr.queue,  # Same queue as transfer!
        self.global_size,
        self.local_size,
        *prepared_args,  # Includes buffers from async writes
    )
```

**Result:** The kernel automatically waits for the buffer transfer because they're in the same command queue.

### CUDA: Stream Ordering

From `src/yapocis/cu/device_manager.py:103-110`:

```python
def write_buffer(self, a: Array, wait: bool = True) -> gpuarray.GPUArray:
    """Write numpy array to PyCUDA GPUArray."""
    buf = self.ensure_buffer(a)

    # Always async transfer to stream
    buf.set_async(a, stream=self.queue)

    if wait:
        # Synchronous: Wait for stream to complete transfer
        self.queue.synchronize()
    # else: Async - transfer continues in background

    return buf
```

**How deferral works:**
1. `async_write=True` → `write(wait=False)` → `set_async()` without `synchronize()`
2. Transfer is enqueued to CUDA stream
3. Kernel execution also uses a stream (from `src/yapocis/cu/kernel.py:157-159`):
   ```python
   def _handle_async_completion(self):
       """Track CUDA stream for async execution."""
       self.kernel_manager.pending_events.append(self.stream)
   ```
4. CUDA stream ordering guarantees kernel waits for transfer

**Result:** Same as OpenCL - GPU ensures kernel doesn't execute until data is ready.

### Metal: No Async Write Support

From `src/yapocis/metallic/device_manager.py:91-100`:

```python
def write_buffer(self, a: Array, wait: bool = True) -> "pymetallic.Buffer":
    """Write numpy array to PyMetallic buffer."""
    # For PyMetallic we create/replace the backing buffer from host bytes
    if not a.flags.c_contiguous:
        a = np.ascontiguousarray(a)

    # NOTE: wait parameter is IGNORED!
    buf = pymetallic.Buffer.from_numpy(self.device, a)  # Synchronous

    buffer_id = id(a)
    self.arrays[buffer_id] = a
    self.buffers[buffer_id] = buf
    return buf
```

**Problem:** `pymetallic.Buffer.from_numpy()` is **always synchronous** - it blocks until the buffer is created and data is transferred. The `wait` parameter is accepted but ignored.

**Result:** `async_write` provides **no benefit** on Metal backend.

## Where async_write is Used

From `src/yapocis/core/kernel_base.py:86-105`:

```python
def handle_in(self, arg, dtype):
    """Handle input parameter: write to GPU."""
    arg = arg.astype(dtype, copy=False)
    buf = self.write(arg, wait=not self.async_write)  # ← Here
    return buf

def handle_inout(self, arg, dtype):
    """Handle input/output parameter: write to GPU and track for return."""
    arg = arg.astype(dtype, copy=False)
    buf = self.write(arg, wait=not self.async_write)  # ← Here
    self.returns.append((arg, buf))
    return buf
```

**Logic:**
- `async_write=False` (default) → `write(wait=True)` → CPU blocks on transfer
- `async_write=True` → `write(wait=False)` → CPU returns immediately

## How to Enable async_write

### Method 1: Global on kernel manager
```python
prog = core.load_program("convolvesep", async_write=True)
# All kernels in this program use async writes
prog.gauss5(image, output)  # Async write for 'image' parameter
```

### Method 2: Via BoundKernels
```python
from yapocis.lib.bound_kernels import BoundKernels

class MyKernels(BoundKernels):
    INTERFACE = "my_interface"
    load_program_config = {"async_write": True}  # ← Set here
```

### Method 3: Per-instance
```python
g = Gaussian(async_write=True)
# Now passed through to kernel manager
```

## Performance Benefit

**When does async_write help?**

✅ **Multiple buffer uploads before kernel execution**
```python
prog = core.load_program("my_kernel", async_write=True)

# Sync version: CPU blocks 3 times (inefficient)
# Async version: All transfers enqueued, kernel waits automatically
result = prog.process(input1, input2, input3, output)
                    # ↑       ↑       ↑
                    # 3 async writes - CPU doesn't wait!
```

✅ **Pipelines with many small kernels**
```python
prog = core.load_program("pipeline", async_write=True)
for i in range(100):
    prog.stage1(data[i], temp)  # Async write 'data[i]'
    prog.stage2(temp, output[i])
# CPU time saved: 100 transfer waits eliminated
```

❌ **Single kernel with one input** - Minimal benefit
❌ **Metal backend** - No benefit (not implemented)
❌ **Kernels with resident buffers** - Buffers already on GPU

## Example: Real-World Usage

From high-level libraries, async_write is typically combined with resident buffers:

```python
# From src/yapocis/lib/median.py
m3 = self.kmgr.median3x3_res
input = image
output = np.empty_like(input)

# Write buffers to GPU (with async_write=True, these don't block)
m3.write(input)
m3.write(output)

# Now execute iterations - buffers already on GPU
with self.kmgr.async_mode():
    for i in range(iterations):
        m3(input, output, resident=(input, output))
        input, output = output, input
```

**Benefit breakdown:**
1. **Initial writes async** - CPU continues to loop setup
2. **Kernel execution async** - GPU pipeline stays full
3. **Result:** Maximum overlap of CPU and GPU work

## Current Limitations

### Metal Backend
The `wait` parameter is accepted but **ignored** in Metal's `write_buffer()`. To fix this, PyMetallic would need to support:

```python
# Hypothetical async API for Metal
def write_buffer(self, a: Array, wait: bool = True):
    command_buffer = self.device.makeCommandBuffer()
    encoder = command_buffer.makeBlitCommandEncoder()
    # ... async blit operations ...
    encoder.endEncoding()
    command_buffer.commit()

    if wait:
        command_buffer.waitUntilCompleted()  # Block
    else:
        return command_buffer  # Return immediately
```

Currently, `Buffer.from_numpy()` is a **synchronous operation** in PyMetallic.

## Summary

**async_write controls CPU blocking on buffer transfers, NOT GPU execution order.**

### OpenCL & CUDA:
- ✅ Command queue/stream ordering ensures kernel waits for data automatically
- ✅ `async_write=True` eliminates CPU blocking on transfers
- ✅ Benefit: CPU can prepare next operations while GPU transfers data

### Metal:
- ❌ `wait` parameter ignored in current implementation
- ❌ All transfers are synchronous (CPU blocks)
- 🔧 Needs PyMetallic enhancement to support async transfers

### The Deferral Mechanism:
**GPU hardware enforces command ordering** - when a transfer and kernel are enqueued to the same queue/stream, the GPU automatically ensures the transfer completes before the kernel executes. The async_write parameter only controls whether the **CPU waits** for the transfer, not whether the **GPU waits** for it (the GPU always waits correctly).
