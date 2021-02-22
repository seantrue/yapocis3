"""
Helpers around the PyOpenCL wrappers for OpenCL
"""
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import os
import sys
import pyopencl
from pyopencl import mem_flags as mf  # @UnusedImport
from yapocis.rpc.buffer_manager import BufferManager
from yapocis.rpc.device import get_device
from yapocis.utils.typing import *

MEMFLAGS = mf.READ_WRITE | mf.COPY_HOST_PTR

YAPOCIS_CACHE = os.environ.get("YAPOCIS_CACHE", None)
if YAPOCIS_CACHE and not os.path.isdir(YAPOCIS_CACHE):
    print(f"YAPOCIS_CACHE={YAPOCIS_CACHE} is not a directory", file=sys.stderr)
    sys.exit(1)


class DeviceManager(BufferManager):
    """
    Coordinates an OpenCL device with a queue, a context, and a buffer manager
    """

    def __init__(self, device_name: Optional[str] = None, autoflatten: bool = True):
        self.device_name, self.device = get_device(device_name=device_name)
        super(DeviceManager, self).__init__(pyopencl.Context([self.device]))
        self.context_meta = {}
        self.calls = 0
        self.runtime = 0.0
        self.autoflatten = autoflatten

    def build(self, source: str) -> pyopencl.Program:
        binary = pyopencl.Program(self.ctx, source)
        program = binary.build(options=["-w"], cache_dir=YAPOCIS_CACHE)
        return program

    def purge(self):
        self.purge_buffers()

    def write(self, a: Array) -> pyopencl.Buffer:
        assert hasattr(a, "dtype")
        return self.write_buffer(a)

    def read(self, a: Array) -> Optional[pyopencl.Buffer]:
        return self.read_buffer(a)

    def ensure(self, a: Array) -> pyopencl.Buffer:
        return self.ensure_buffer(a)


def get_device_manager(device_name: Optional[str] = None) -> DeviceManager:
    """
    Takes the identifier for an OpenCL device
    and returns a DeviceManager with buffer management .
    """
    return DeviceManager(device_name)

# Todo: Not thread safe: allocate a new Program and Kernels per thread
