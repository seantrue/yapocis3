"""
Helpers around the PyOpenCL wrappers for OpenCL
"""
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

import os
import sys

import pyopencl
import yaml
from pyopencl import mem_flags as mf  # @UnusedImport

from yapocis.ocl.buffer_manager import BufferManager
from yapocis.ocl.device import get_device
from yapocis.utils.stereo_typing import *

MEMFLAGS = mf.READ_WRITE | mf.COPY_HOST_PTR

YAPOCIS_CACHE = os.environ.get("YAPOCIS_CACHE", None)
if YAPOCIS_CACHE and not os.path.isdir(YAPOCIS_CACHE):
    print(f"YAPOCIS_CACHE={YAPOCIS_CACHE} is not a directory", file=sys.stderr)
    sys.exit(1)


class DeviceManager(BufferManager):
    """
    Coordinates an OpenCL device with a queue, a context, and a buffer manager
    """

    def __init__(self, device_name: Optional[str] = None, timed: bool = False, **kwargs):
        assert device_name is None or isinstance(device_name, (str, int, bytes)), device_name
        self.device_name, self.device = get_device(device_name=device_name)
        try:
            super(DeviceManager, self).__init__(pyopencl.Context([self.device]), None, timed=timed, **kwargs)
        except RuntimeError as e:
            print(f"Failed to initialize `{self.device_name}', trying default")
            self.device_name, self.device = get_device()
            super(DeviceManager, self).__init__(pyopencl.Context([self.device]), None, timed=timed, **kwargs)
        self.context_meta = {}

    def build(self, source: str, options=None) -> pyopencl.Program:
        binary = pyopencl.Program(self.ctx, source)
        options =  ["-w","-cl-fast-relaxed-math", "-cl-mad-enable","-cl-no-signed-zeros","-cl-finite-math-only"] if options is None else options

        program = binary.build(options=options, cache_dir=YAPOCIS_CACHE)
        return program

    def purge(self):
        self.purge_buffers()

    def write(self, a: Array, wait=True) -> pyopencl.Buffer:
        assert hasattr(a, "dtype")
        return self.write_buffer(a, wait=wait)

    def read(self, a: Array) -> Optional[pyopencl.Buffer]:
        return self.read_buffer(a)

    def ensure(self, a: Array) -> pyopencl.Buffer:
        return self.ensure_buffer(a)


_device_managers: Dict[str, DeviceManager] = {}


def get_device_manager(device_name: Optional[str] = None) -> DeviceManager:
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
