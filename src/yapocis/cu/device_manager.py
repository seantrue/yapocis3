"""
Helpers around the PyCUDA wrappers for CUDA
"""
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.

import atexit
import os
import sys

import pycuda.driver as cuda
cuda.init()  # Initialize CUDA driver (but don't auto-create context)
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import yaml

from yapocis.core.device_mgr_base import DeviceManagerBase
from yapocis.cu.device import get_device
from yapocis.utils.stereo_typing import Array

YAPOCIS_CACHE = os.environ.get("YAPOCIS_CACHE", None)
if YAPOCIS_CACHE and not os.path.isdir(YAPOCIS_CACHE):
    print(f"YAPOCIS_CACHE={YAPOCIS_CACHE} is not a directory", file=sys.stderr)
    sys.exit(1)


class DeviceManager(DeviceManagerBase):
    """
    CUDA device manager - coordinates device, context, stream, and buffer management.

    Inherits from DeviceManagerBase (which includes buffer management functionality).
    """

    def __init__(self, device_name: str | None = None, timed: bool = False, **kwargs):
        """
        Initialize CUDA device manager.

        Args:
            device_name: Device identifier (None for default)
            timed: Enable performance timing
            **kwargs: Additional arguments for DeviceManagerBase
        """
        assert device_name is None or isinstance(device_name, (str, int, bytes)), device_name
        device_name_str, device = get_device(device_name=device_name)

        # Create context and stream
        try:
            ctx = device.retain_primary_context()
            ctx.push()  # Make this context current
            stream = cuda.Stream()
            super().__init__(ctx, stream, device_name_str, device, timed=timed, **kwargs)
        except RuntimeError as e:
            print(f"Failed to initialize `{device_name_str}', trying default")
            device_name_str, device = get_device()
            ctx = device.retain_primary_context()
            ctx.push()  # Make this context current
            stream = cuda.Stream()
            super().__init__(ctx, stream, device_name_str, device, timed=timed, **kwargs)

    def build(self, source: str, options: list[str] | None = None) -> SourceModule:
        """
        Compile CUDA source code.

        Args:
            source: CUDA kernel source code
            options: Compiler options (uses fast math defaults if None)

        Returns:
            Compiled SourceModule
        """
        # Default CUDA compilation options
        default_options = [
            "--use_fast_math",
            "-Xcudafe", "--diag_suppress=177",  # Suppress "function declared but never referenced"
            "-Xcudafe", "--diag_suppress=68",   # Suppress "integer conversion resulted in a change of sign"
            "-Xcudafe", "--diag_suppress=550"   # Suppress "variable was set but never used"
        ]
        options = default_options if options is None else options

        # Compile the source module (no_extern_c=True to allow C++ templates)
        module = SourceModule(source, options=options, cache_dir=YAPOCIS_CACHE, no_extern_c=True)
        return module

    # Implement abstract methods from DeviceManagerBase
    def make_buffer(self, a: Array) -> gpuarray.GPUArray:
        """Create PyCUDA GPUArray from numpy array."""
        if not a.data.contiguous:
            raise RuntimeError(f"Array {id(a)} {a.shape} is not contiguous")
        # Create a GPUArray from the numpy array
        buf = gpuarray.to_gpu_async(a, stream=self.queue)
        buffer_id = id(a)
        self.arrays[buffer_id] = a
        self.buffers[buffer_id] = buf
        return buf

    def read_buffer(self, a: Array) -> Array:
        """Read PyCUDA GPUArray back to numpy array."""
        buf = self.find_buffer(a, self.READ)
        # Copy from GPU to host
        buf.get_async(self.queue, ary=a)
        self.queue.synchronize()
        return a

    def write_buffer(self, a: Array, wait: bool = True) -> gpuarray.GPUArray:
        """Write numpy array to PyCUDA GPUArray."""
        buf = self.ensure_buffer(a)
        # Copy from host to GPU
        buf.set_async(a, stream=self.queue)
        if wait:
            self.queue.synchronize()
        return buf

    def find_buffer(self, a: Array, op: int) -> gpuarray.GPUArray | None:
        """Find existing PyCUDA GPUArray for array."""
        assert op in (self.READ, self.WRITE)
        buffer_id = id(a)
        havea = buffer_id in self.buffers and buffer_id in self.arrays
        # Complete match, easy decision
        if havea:
            self.hits += 1
            return self.buffers[buffer_id]
        else:
            self.misses += 1
        # No match at all, also easy.
        if not havea:
            # Reading an array back with no matching buffer is fatal
            if op == self.READ:
                raise ValueError(
                    "Array not in yapocis management, you may have written to it, or be using an assigned or .copy"
                )
            return None
        raise RuntimeError(f"Failed to find a buffer for {op} {a}")

    def wait_events(self, events: list) -> None:
        """
        Wait for a list of CUDA streams to complete.

        Args:
            events: List of CUDA streams to wait for
        """
        if events:
            # CUDA streams have synchronize() method
            # Collect unique streams
            unique_streams = set()
            for event in events:
                if hasattr(event, 'synchronize'):
                    unique_streams.add(event)

            # Synchronize each unique stream
            for stream in unique_streams:
                stream.synchronize()

    def _free_buffer(self, buffer: gpuarray.GPUArray):
        """
        Prepare PyCUDA GPUArray for garbage collection.

        We don't call gpudata.free() explicitly because it can cause
        "illegal memory access" errors if the context is no longer valid.
        Instead, we let PyCUDA's own cleanup handle it during GC.

        Args:
            buffer: PyCUDA GPUArray to mark for cleanup
        """
        # Just ensure buffer is dereferenced - PyCUDA will clean up during GC
        pass

    def _close(self):
        """
        CUDA-specific cleanup operations.

        Synchronize stream and prepare for context cleanup.
        """
        try:
            # Synchronize stream to finish all pending operations
            if hasattr(self, 'queue') and self.queue is not None:
                self.queue.synchronize()
        except cuda.Error:
            # Stream may already be invalid - ignore
            pass

        # Clear buffer references to allow GPU memory to be freed
        self.buffers.clear()
        self.arrays = {}

    def __del__(self):
        """
        Clean up context on deletion.

        Most cleanup is handled by atexit handler. This is a fallback
        in case __del__ is called before atexit (rare but possible).
        """
        try:
            # Skip if already cleaned up by atexit handler
            if not hasattr(self, 'ctx') or self.ctx is None:
                return

            # Clear buffers
            if hasattr(self, 'buffers'):
                self.buffers.clear()

            # Pop context as fallback
            try:
                self.ctx.pop()
                self.ctx = None
            except (cuda.Error, AttributeError, RuntimeError):
                # Context may already be invalid
                pass
        except Exception:
            # Silently ignore all errors during __del__
            pass

_device_managers: dict[str, DeviceManager] = {}

def _cleanup_device_managers():
    """
    Cleanup all device managers at program exit.

    Registered as atexit handler to ensure proper resource cleanup.
    Only performs cleanup if explicitly enabled to avoid pytest issues.
    """
    # Only do aggressive cleanup if explicitly requested
    # During pytest, this cleanup can corrupt contexts for other tests
    if os.environ.get('YAPOCIS_ATEXIT_CLEANUP', '').lower() not in ('1', 'true', 'yes'):
        return

    for device_name, dmgr in list(_device_managers.items()):
        try:
            # Purge buffers before exit
            dmgr.purge()

            # Clear buffer and module references
            dmgr.buffers.clear()

            # Pop context from stack
            if hasattr(dmgr, 'ctx') and dmgr.ctx is not None:
                try:
                    dmgr.ctx.pop()
                    dmgr.ctx = None  # Mark as cleaned up
                except (cuda.Error, AttributeError, RuntimeError):
                    # Context may already be invalid
                    pass
        except Exception:
            # Ignore errors during cleanup
            pass

# Register cleanup handler
atexit.register(_cleanup_device_managers)

def get_device_manager(device_name: str | None = None) -> DeviceManager:
    """
    Takes the identifier for a CUDA device
    and returns a DeviceManager with buffer management.

    Device managers are cached by device name to maintain context stability.
    CUDA contexts must remain alive for compiled kernels to work, so we
    keep one DeviceManager per device for the lifetime of the process.
    """
    # Normalize device_name to string for cache key
    if device_name is None:
        cache_key = "default"
    elif isinstance(device_name, int):
        cache_key = f"device_{device_name}"
    else:
        cache_key = str(device_name)

    # Return cached manager or create new one
    if cache_key not in _device_managers:
        _device_managers[cache_key] = DeviceManager(device_name)

    return _device_managers[cache_key]

def display_stats():
    device_stats = {dn: dmgr.stats for dn, dmgr in _device_managers.items()}
    print(yaml.safe_dump(device_stats))

def display_snapshots():
    print()
    print("Snapshots")
    for dn, dmgr in _device_managers.items():
        print(f"Device: {dn}")
        print(yaml.safe_dump(dmgr.snapshots))
