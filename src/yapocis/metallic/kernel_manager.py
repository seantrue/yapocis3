from __future__ import annotations

"""
KernelManager for Metal (PyMetallic) backend.

Renders Mako templates to Metal shader source, builds a program via PyMetallic,
and exposes callable kernel functions mirroring the OpenCL KernelManager API.

This implementation uses KernelManagerBase for common functionality.
"""

import os

try:
    import pymetallic as pm  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyMetallic is required for the metallic backend") from e

from yapocis.core.kmgr_base import KernelManagerBase
from yapocis.metallic.device_manager import get_device_manager

# Runtime selection between Python and Cython implementations
_use_cython = os.environ.get("YAPOCIS_USE_CYTHON", "0").lower() in ("1", "true", "yes")

if _use_cython:
    try:
        from yapocis.metallic.kernel_cy import KernelCy as Kernel

        print("Using Cython-optimized Metal kernel")
    except ImportError:
        from yapocis.metallic.kernel import Kernel

        print(
            "Warning: YAPOCIS_USE_CYTHON=1 but Cython extension not built, falling back to Python"
        )
else:
    from yapocis.metallic.kernel import Kernel


class KernelManager(KernelManagerBase):
    """
    Metal backend kernel manager using PyMetallic.

    Metal-specific behaviors:
    - Prefers metalkernels/*.mako, falls back to kernels/*.mako (OCL templates)
    - Uses pymetallic.Kernel for compilation
    - Includes 'backend=metal' in snapshots
    """

    _backend_name = "metal"
    _kernel_class = Kernel

    def get_device_manager(self, device_name: str | int | None):
        """Return Metal device manager."""
        return get_device_manager(device_name=device_name)

    def get_kernel(self, program, kernel_name: str):
        """
        Metal doesn't use a traditional 'program' object.

        Instead, pymetallic.Kernel compiles each kernel individually from source.
        We store the source in self.source and create kernels on demand.
        """
        # PyMetallic creates kernels individually with source
        return pm.Kernel(self.dmgr.device, source=self.source, func=kernel_name)

    # Template resolution (kernels/metal, then the kernels/ocl fallback) is the
    # module-level ``yapocis.core.kmgr_base.find_kernel_template``.

    def snapshot(self, **kwargs):
        """Add backend='metal' to snapshot metadata."""
        d = super().snapshot(backend="metal")
        return d

    def stats(self):
        """Add backend='metal' to stats."""
        d = super().stats()
        d["backend"] = "metal"
        return d


# Module-level convenience functions for API compatibility
import_program = KernelManager.import_program
load_program = KernelManager.load_program
