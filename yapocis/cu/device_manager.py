"""
Helpers around the PyCUDA wrappers for CUDA
"""
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.

import os
import sys

import pycuda.driver as cuda
cuda.init()  # Initialize CUDA driver (but don't auto-create context)
from pycuda.compiler import SourceModule
import yaml

from yapocis.cu.buffer_manager import BufferManager
from yapocis.cu.device import get_device
from yapocis.utils.stereo_typing import *

YAPOCIS_CACHE = os.environ.get("YAPOCIS_CACHE", None)
if YAPOCIS_CACHE and not os.path.isdir(YAPOCIS_CACHE):
    print(f"YAPOCIS_CACHE={YAPOCIS_CACHE} is not a directory", file=sys.stderr)
    sys.exit(1)


class DeviceManager(BufferManager):
    """
    Coordinates a CUDA device with a stream, a context, and a buffer manager
    """

    def __init__(self, device_name: Optional[str] = None, timed: bool = False, **kwargs):
        assert device_name is None or isinstance(device_name, (str, int, bytes)), device_name
        self.device_name, self.device = get_device(device_name=device_name)

        # Create context and stream
        try:
            self.ctx = self.device.retain_primary_context()
            self.ctx.push()  # Make this context current
            self.stream = cuda.Stream()
            super(DeviceManager, self).__init__(self.ctx, self.stream, timed=timed, **kwargs)
        except RuntimeError as e:
            print(f"Failed to initialize `{self.device_name}', trying default")
            self.device_name, self.device = get_device()
            self.ctx = self.device.retain_primary_context()
            self.ctx.push()  # Make this context current
            self.stream = cuda.Stream()
            super(DeviceManager, self).__init__(self.ctx, self.stream, timed=timed, **kwargs)
        self.context_meta = {}

    def build(self, source: str, options=None) -> SourceModule:
        """Compile CUDA source code"""
        # Default CUDA compilation options
        default_options = ["--use_fast_math"]
        options = default_options if options is None else options

        # Compile the source module (no_extern_c=True to allow C++ templates)
        module = SourceModule(source, options=options, cache_dir=YAPOCIS_CACHE, no_extern_c=True)
        return module

    def purge(self):
        self.purge_buffers()

    def write(self, a: Array, wait=True):
        assert hasattr(a, "dtype")
        return self.write_buffer(a, wait=wait)

    def read(self, a: Array):
        return self.read_buffer(a)

    def ensure(self, a: Array):
        return self.ensure_buffer(a)

    def __del__(self):
        """Clean up context on deletion"""
        try:
            if hasattr(self, 'ctx') and self.ctx:
                self.ctx.pop()
        except:
            pass


_device_managers: Dict[str, DeviceManager] = {}


def get_device_manager(device_name: Optional[str] = None) -> DeviceManager:
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
