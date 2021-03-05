#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.
from yapocis.utils.typing import *
from yapocis.rpc.kernel_manager import load_program
class BoundKernels:
    INTERFACE = None
    def __init__(self, device_name:Optional[str]=None, **kwargs):
        self.kmgr = load_program(self.INTERFACE, device_name=device_name, **kwargs)

    def snapshot(self, **kwargs):
        self.kmgr.purge()
        self.kmgr.snapshot(**kwargs)
        self.kmgr.reset()

