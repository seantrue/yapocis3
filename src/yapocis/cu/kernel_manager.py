#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.

"""
KernelManager for CUDA backend.

Renders Mako templates to CUDA source, builds a program via PyCUDA,
and exposes callable kernel functions.

This implementation uses KernelManagerBase for common functionality.
"""

from yapocis.core.kmgr_base import KernelManagerBase
from yapocis.cu.device_manager import get_device_manager
from yapocis.cu.kernel import Kernel


class KernelManager(KernelManagerBase):
    """
    CUDA backend kernel manager using PyCUDA.

    CUDA-specific behaviors:
    - Uses cukernels/*.mako templates
    - Uses program.get_function() for kernel extraction
    - Source caching already handled by base class
    """

    _backend_name = "cuda"
    _kernel_class = Kernel

    def get_device_manager(self, device_name: str | int | None):
        """Return CUDA device manager."""
        return get_device_manager(device_name=device_name)

    def get_kernel(self, program, kernel_name: str):
        """
        Extract kernel from compiled CUDA program.

        CUDA uses get_function() to access kernels from program object.

        Args:
            program: Compiled pycuda.driver.Module
            kernel_name: Name of kernel function

        Returns:
            pycuda.driver.Function object
        """
        return program.get_function(kernel_name)

    def snapshot(self, **kwargs):
        """Add backend='metal' to snapshot metadata."""
        d = super().snapshot(backend="cuda")
        return d

    def stats(self):
        """Add backend='metal' to stats."""
        d = super().stats()
        d["backend"] = "cuda"
        return d


# Module-level convenience functions for API compatibility
import_program = KernelManager.import_program
load_program = KernelManager.load_program
