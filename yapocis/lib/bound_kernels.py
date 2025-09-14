#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.
from yapocis.core.manager import load_program
from yapocis.utils.stereo_typing import *


class BoundKernels:
    """Lightweight wrapper that binds an interface to a backend KernelManager.

    Notes
    - Honors `YAPOCIS_BACKEND` automatically, or accept explicit `backend`.
    - `device_name` may be a device index or name string.
    - Extra `kwargs` are forwarded to the backend loader (e.g., Mako vars,
      `debug`, `options`, `async_write`).
    """

    INTERFACE = None
    load_program_config = {}

    def __init__(self,
                 device_name: Optional[EngineID] = None,
                 backend: Optional[str] = None,
                 **kwargs):
        assert device_name is None or isinstance(device_name, (int, str, bytes)), device_name
        # Pass backend through so callers can force Metal/CUDA/OCL per instance.
        self.kmgr = load_program(
            self.INTERFACE,
            backend=backend,
            device_name=device_name,
            **self.load_program_config,
            **kwargs,
        )

    def __getattr__(self, attr: str) -> Callable:
        "Any unbound attribute is checked to see if it is a kernel"
        if attr not in self.kmgr.callable:
            print("BoundKernel %s not found, %s available" % (attr, list(self.kmgr.callable.keys())))
            raise KeyError(attr)
        return self.kmgr.callable[attr]


    def write(self, a):
        self.kmgr.write(a)

    def read(self, a):
        return self.kmgr.read(a)

    def snapshot(self, **kwargs):
        self.kmgr.purge()
        self.kmgr.snapshot(**kwargs)
        self.kmgr.reset()

    def purge(self):
        self.kmgr.purge()
