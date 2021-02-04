#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import os
import pyopencl
from pyopencl import mem_flags as mf  # @UnusedImport
from yapocis.rpc.buffers import BufferManager
from yapocis.lib.yapocis_types import *
MEMFLAGS = mf.READ_WRITE | mf.COPY_HOST_PTR


from pyopencl import get_platforms
def get_available():
    platforms = get_platforms()
    devices = {}
    for platform in platforms:
        for device in platform.get_devices():
            devices[device.name] = device
            if len(platforms) > 1:
                devices[f"{platform.name}:{device.name}"] = device
    return devices

def get_yapocis_device(engine_name:Optional[str] = None):
    engine_name = os.environ.get("YAPOCIS_ENGINE",None) if engine_name is None else engine_name
    devices = get_available()
    if not devices:
        raise RuntimeError("No OpenCL devices found")
    device_names = list(devices.keys())
    if engine_name is None:
        engine_name = device_names[0]
        print(f"Defaulting to {engine_name}")
    if engine_name not in devices:
        try:
            engine_index = int(engine_name)
        except ValueError:
            raise RuntimeError(f"{engine_name} not an available device {device_names}")
        return get_yapocis_device(device_names[engine_index])
    return engine_name, devices[engine_name]



class Engine:
    def __init__(self, engine_name: Optional[str] = None, autoflatten: bool = True):
        self.engine_name, self.device = get_yapocis_device(engine_name=engine_name)
        self.ctx = pyopencl.Context([self.device])
        self.queue = pyopencl.CommandQueue(self.ctx)
        self.bmgr = BufferManager(context=self.ctx, queue=self.queue)
        self.context_meta = {}
        self.calls = 0
        self.runtime = 0.0
        self.autoflatten = autoflatten

    def build(self, source: str) -> pyopencl.Program:
        binary = pyopencl.Program(self.ctx, source)
        program = binary.build(options=["-w"])
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


def get_engine(engine_name: Optional[str] = None) -> Engine:
    """Takes the identifier for an OpenCL engine (CPU_ENGINE, GPU_ENGINE)
    and returns a Engine with buffer management ."""
    return Engine(engine_name)

# Todo: Not thread safe: allocate a new Program and Kernels per thread
