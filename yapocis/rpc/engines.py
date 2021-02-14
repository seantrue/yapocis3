"""
Helpers around the PyOpenCL wrappers for OpenCL
"""
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import os
import sys
import pyopencl
from pyopencl import mem_flags as mf  # @UnusedImport
from yapocis.rpc.buffers import BufferManager
from yapocis.utils.typing import *

MEMFLAGS = mf.READ_WRITE | mf.COPY_HOST_PTR

YAPOCIS_CACHE = os.environ.get("YAPOCIS_CACHE", None)
if YAPOCIS_CACHE and not os.path.isdir(YAPOCIS_CACHE):
    print(f"YAPOCIS_CACHE={YAPOCIS_CACHE} is not a directory", file=sys.stderr)
    sys.exit(1)
from pyopencl import get_platforms


def get_available(aliases:bool=True) -> Dict[str, "Device"]:
    """
    Get the names of the available OpenCL platforms and devices. Name is device name,
    with an alias of platform:device name.

    Returns: a dictionary mapping names to devices, suitable for use with get_yapocis_device

    """
    platforms = get_platforms()
    devices = {}
    idevice = 0
    for platform in platforms:
        for device in platform.get_devices():
            devices[device.name] = device
            if aliases:
                devices[idevice] = device
            idevice += 1
            if len(platforms) > 1 and aliases:
                devices[f"{platform.name}:{device.name}"] = device
    return devices


def get_yapocis_device(device_name: Optional[str] = None) -> "Device":
    """
    Get a device that is an exact or approximate match for device name, which can also be an numeric index,
    and if None, will be the first device found.

    Args:
        device_name: if present, either a name or string/int that can be index into the device dictionary

    Returns: an OpenCL device as wrapped by PyOpenCL

    """
    device_name = os.environ.get("YAPOCIS_ENGINE", None) if device_name is None else device_name
    devices = get_available()
    if not devices:
        raise RuntimeError("No OpenCL devices found")
    device_names = list(devices.keys())
    if device_name is None:
        device_name = device_names[0]
        print(f"Defaulting to {device_name}")
    if device_name not in devices:
        try:
            engine_index = int(device_name)
        except ValueError:
            raise RuntimeError(f"{device_name} not an available device {device_names}")
        return get_yapocis_device(device_names[engine_index])
    return device_name, devices[device_name]


class Engine:
    """
    Coordinates an OpenCL device with a queue, a context, and a buffer manager
    """

    def __init__(self, device_name: Optional[str] = None, autoflatten: bool = True):
        self.device_name, self.device = get_yapocis_device(device_name=device_name)
        self.ctx = pyopencl.Context([self.device])
        self.queue = pyopencl.CommandQueue(self.ctx)
        self.bmgr = BufferManager(context=self.ctx, queue=self.queue)
        self.context_meta = {}
        self.calls = 0
        self.runtime = 0.0
        self.autoflatten = autoflatten

    def build(self, source: str) -> pyopencl.Program:
        binary = pyopencl.Program(self.ctx, source)
        program = binary.build(options=["-w"], cache_dir=YAPOCIS_CACHE)
        return program

    def purge(self):
        self.bmgr.purge_buffers()

    def write(self, a: Array) -> pyopencl.Buffer:
        assert hasattr(a, "dtype")
        return self.bmgr.write_buffer(a)

    def read(self, a: Array) -> Optional[pyopencl.Buffer]:
        return self.bmgr.read_buffer(a)

    def ensure(self, a: Array) -> pyopencl.Buffer:
        return self.bmgr.ensure_buffer(a)


def get_engine(device_name: Optional[str] = None) -> Engine:
    """
    Takes the identifier for an OpenCL engine (CPU_ENGINE, GPU_ENGINE)
    and returns a Engine with buffer management .
    """
    return Engine(device_name)

# Todo: Not thread safe: allocate a new Program and Kernels per thread
