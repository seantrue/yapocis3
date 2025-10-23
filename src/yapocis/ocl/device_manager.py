"""
Helpers around the PyOpenCL wrappers for OpenCL
"""
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

import os
import sys

import pyopencl
import yaml
from pyopencl import mem_flags as mf  # @UnusedImport

from yapocis.core.device_mgr_base import DeviceManagerBase
from yapocis.ocl.device import get_device
from yapocis.utils.stereo_typing import Array

MEMFLAGS = mf.READ_WRITE | mf.COPY_HOST_PTR

YAPOCIS_CACHE = os.environ.get("YAPOCIS_CACHE", None)
if YAPOCIS_CACHE and not os.path.isdir(YAPOCIS_CACHE):
    print(f"YAPOCIS_CACHE={YAPOCIS_CACHE} is not a directory", file=sys.stderr)
    sys.exit(1)


class DeviceManager(DeviceManagerBase):
    """
    OpenCL device manager - coordinates device, context, queue, and buffer management.

    Inherits from DeviceManagerBase (which includes buffer management functionality).
    """

    MEMFLAGS = mf.READ_WRITE | mf.COPY_HOST_PTR

    def __init__(self, device_name: str | None = None, timed: bool = False, **kwargs):
        """
        Initialize OpenCL device manager.

        Args:
            device_name: Device identifier (None for default)
            timed: Enable performance timing
            **kwargs: Additional arguments for DeviceManagerBase
        """
        assert device_name is None or isinstance(device_name, (str, int, bytes)), device_name
        device_name_str, device = get_device(device_name=device_name)
        try:
            context = pyopencl.Context([device])
            queue = pyopencl.CommandQueue(context)
            super().__init__(context, queue, device_name_str, device, timed=timed, **kwargs)
        except RuntimeError as e:
            print(f"Failed to initialize `{device_name_str}', trying default")
            device_name_str, device = get_device()
            context = pyopencl.Context([device])
            queue = pyopencl.CommandQueue(context)
            super().__init__(context, queue, device_name_str, device, timed=timed, **kwargs)

    def build(self, source: str, options: list[str] | None = None) -> pyopencl.Program:
        """
        Compile OpenCL source code.

        Args:
            source: OpenCL kernel source code
            options: Compiler options (uses fast math defaults if None)

        Returns:
            Compiled pyopencl.Program
        """
        binary = pyopencl.Program(self.ctx, source)
        if options is None:
            options = ["-w", "-cl-fast-relaxed-math", "-cl-mad-enable",
                      "-cl-no-signed-zeros", "-cl-finite-math-only"]
        program = binary.build(options=options, cache_dir=YAPOCIS_CACHE)
        return program

    # Implement abstract methods from DeviceManagerBase
    def make_buffer(self, a: Array) -> pyopencl.Buffer:
        """Create PyOpenCL buffer from numpy array."""
        if not a.data.contiguous:
            raise RuntimeError(f"Array {id(a)} {a.shape} is not contiguous")
        buf = pyopencl.Buffer(self.ctx, self.MEMFLAGS, hostbuf=a)
        buffer_id = id(a)
        self.arrays[buffer_id] = a
        self.buffers[buffer_id] = buf
        return buf

    def read_buffer(self, a: Array) -> Array:
        """Read PyOpenCL buffer back to numpy array."""
        buf = self.find_buffer(a, self.READ)
        shape = a.shape
        strides = a.strides
        a.shape = (a.size,)
        a.strides = (strides[-1],)
        pyopencl.enqueue_copy(self.queue, a, buf).wait()
        a.shape = shape
        a.strides = strides
        return a

    def write_buffer(self, a: Array, wait: bool = True) -> pyopencl.Buffer:
        """Write numpy array to PyOpenCL buffer."""
        buf = self.ensure_buffer(a)
        shape = a.shape
        strides = a.strides
        a.shape = (a.size,)
        a.strides = (strides[-1],)
        if wait:
            pyopencl.enqueue_copy(self.queue, buf, a).wait()
        else:
            pyopencl.enqueue_copy(self.queue, buf, a)
        a.shape = shape
        a.strides = strides
        return buf

    def find_buffer(self, a: Array, op: int) -> pyopencl.Buffer | None:
        """Find existing PyOpenCL buffer for array."""
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
                    "Array not in yapocis management, you may have written to it, or be using an assigned or .copy")
            return None
        raise RuntimeError(f"Failed to find a buffer for {op} {a}")

_device_managers: dict[str, DeviceManager] = {}

def get_device_manager(device_name: str | None = None) -> DeviceManager:
    """
    Takes the identifier for an OpenCL device
    and returns a DeviceManager with buffer management
    """
    try:
        dmgr = _device_managers[device_name]
    except KeyError:
        _device_managers[device_name] = dmgr = DeviceManager(device_name)
    return dmgr

def display_stats():
    device_stats = {dn: dmgr.stats for dn, dmgr in _device_managers.items()}
    print(yaml.safe_dump(device_stats))

def display_snapshots():
    print()
    print("Snapshots")
    for dn, dmgr in _device_managers.items():
        print(f"Device: {dn}")
        print(yaml.safe_dump(dmgr.snapshots))
