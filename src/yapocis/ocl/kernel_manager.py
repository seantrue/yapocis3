#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

"""
KernelManager for OpenCL backend.

Renders Mako templates to OpenCL source, builds a program via PyOpenCL,
and exposes callable kernel functions.

This implementation uses KernelManagerBase for common functionality.
"""

import pyopencl

from yapocis.core.kmgr_base import KernelManagerBase
from yapocis.ocl.device_manager import get_device_manager
from yapocis.ocl.kernel import Kernel


class KernelManager(KernelManagerBase):
    """
    OpenCL backend kernel manager using PyOpenCL.

    OpenCL-specific behaviors:
    - Uses kernels/*.mako templates
    - Stores pyopencl.mem_flags for buffer allocation
    - Uses attribute access for kernel extraction: getattr(program, name)
    """

    _backend_name = "ocl"
    _kernel_class = Kernel

    def __init__(self, *args, **kwargs):
        """
        Initialize OCL kernel manager.

        Extends base __init__ to store mem_flags for OCL Kernel class.
        """
        super().__init__(*args, **kwargs)
        # OCL Kernel class needs access to mem_flags
        self.mf = pyopencl.mem_flags

    def get_device_manager(self, device_name: str | int | None):
        """Return OpenCL device manager."""
        return get_device_manager(device_name=device_name)

    def get_kernel(self, program, kernel_name: str):
        """
        Extract kernel from compiled OpenCL program with caching.

        Creates and caches kernel instances to avoid the overhead of repeated
        kernel retrieval. PyOpenCL's getattr on programs creates new kernel
        objects each time, which is expensive for frequently-called kernels.

        Args:
            program: Compiled pyopencl.Program
            kernel_name: Name of kernel function

        Returns:
            pyopencl.Kernel object (cached instance)
        """
        # Initialize kernel cache on first use
        if not hasattr(program, "_kernel_cache"):
            program._kernel_cache = {}

        # Return cached kernel or create and cache new one
        if kernel_name not in program._kernel_cache:
            program._kernel_cache[kernel_name] = pyopencl.Kernel(program, kernel_name)

        return program._kernel_cache[kernel_name]

    def snapshot(self, **kwargs):
        """Add backend='metal' to snapshot metadata."""
        d = super().snapshot(backend="ocl")
        return d

    def stats(self):
        """Add backend='metal' to stats."""
        d = super().stats()
        d["backend"] = "ocl"
        return d


# Module-level convenience functions for API compatibility
import_program = KernelManager.import_program
load_program = KernelManager.load_program
