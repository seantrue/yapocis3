#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.
from yapocis.utils.stereo_typing import *
from yapocis.ocl.kernel_manager import load_program
class BoundKernels:
    INTERFACE = None
    def __init__(self, device_name:Optional[str]=None, **kwargs):
        self.kmgr = load_program(self.INTERFACE, device_name=device_name, **kwargs)

    def __getattr__(self, attr: str) -> Callable:
        "Any unbound attribute is checked to see if it is a kernel"
        if not attr in self.kmgr.callable:
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

