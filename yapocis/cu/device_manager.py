"""
Helpers around the PyCUDA wrappers for CUDA
"""
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.

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

    Inherits from DeviceManagerBase (which extends BufferManagerBase).
    """

    def __init__(self, device_name: str | None = None, timed: bool = False, **kwargs):
        """
        Initialize CUDA device manager.

        Args:
            device_name: Device identifier (None for default)
            timed: Enable performance timing
            **kwargs: Additional arguments for BufferManagerBase
        """
        assert device_name is None or isinstance(device_name, (str, int, bytes)), device_name
        device_name_str, device = get_device(device_name=device_name)

        # Create context and stream
        try:
            ctx = device.retain_primary_context()
            ctx.push()  # Make this context current
            self.stream = cuda.Stream()
            super().__init__(ctx, self.stream, device_name_str, device, timed=timed, **kwargs)
        except RuntimeError as e:
            print(f"Failed to initialize `{device_name_str}', trying default")
            device_name_str, device = get_device()
            ctx = device.retain_primary_context()
            ctx.push()  # Make this context current
            self.stream = cuda.Stream()
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
        default_options = ["--use_fast_math"]
        options = default_options if options is None else options

        # Compile the source module (no_extern_c=True to allow C++ templates)
        module = SourceModule(source, options=options, cache_dir=YAPOCIS_CACHE, no_extern_c=True)
        return module

    # Implement abstract methods from BufferManagerBase
    def make_buffer(self, a: Array) -> gpuarray.GPUArray:
        """Create PyCUDA GPUArray from numpy array."""
        if not a.data.contiguous:
            raise RuntimeError(f"Array {id(a)} {a.shape} is not contiguous")
        # Create a GPUArray from the numpy array
        buf = gpuarray.to_gpu_async(a, stream=self.stream)
        buffer_id = id(a)
        self.arrays[buffer_id] = a
        self.buffers[buffer_id] = buf
        return buf

    def read_buffer(self, a: Array) -> Array:
        """Read PyCUDA GPUArray back to numpy array."""
        buf = self.find_buffer(a, self.READ)
        # Copy from GPU to host
        buf.get_async(self.stream, ary=a)
        self.stream.synchronize()
        return a

    def write_buffer(self, a: Array, wait: bool = True) -> gpuarray.GPUArray:
        """Write numpy array to PyCUDA GPUArray."""
        buf = self.ensure_buffer(a)
        # Copy from host to GPU
        buf.set_async(a, stream=self.stream)
        if wait:
            self.stream.synchronize()
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

    def __del__(self):
        """Clean up context on deletion"""
        try:
            if hasattr(self, 'ctx') and self.ctx:
                self.ctx.pop()
        except:
            pass

_device_managers: dict[str, DeviceManager] = {}

def get_device_manager(device_name: str | None = None) -> DeviceManager:
    """
    Takes the identifier for a CUDA device
    and returns a DeviceManager with buffer management
    """
    # Disable caching - create fresh DeviceManager each time
    # This avoids context issues when switching between backends
    return DeviceManager(device_name)

def display_stats():
    device_stats = {dn: dmgr.stats for dn, dmgr in _device_managers.items()}
    print(yaml.safe_dump(device_stats))

def display_snapshots():
    print()
    print("Snapshots")
    for dn, dmgr in _device_managers.items():
        print(f"Device: {dn}")
        print(yaml.safe_dump(dmgr.snapshots))

def cleanup_contexts():
    """Clean up all CUDA contexts on exit"""
    for dmgr in _device_managers.values():
        try:
            if hasattr(dmgr, 'ctx') and dmgr.ctx:
                dmgr.ctx.pop()
        except:
            pass

import atexit
atexit.register(cleanup_contexts)
