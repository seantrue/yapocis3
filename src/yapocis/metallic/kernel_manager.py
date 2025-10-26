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
from yapocis.metallic.kernel import Kernel
from yapocis.utils.internals import get_yapocis_dir


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

    def find_kernel_template(self, base_filename: str) -> str:
        """
        Metal-specific template resolution with OCL fallback.

        Looks for templates in:
        1. kernels/metal/*.mako (Metal-specific)
        2. kernels/ocl/*.mako (OCL fallback)

        Args:
            base_filename: Interface name (e.g., "mandelbrot")

        Returns:
            Full path to template file

        Raises:
            FileNotFoundError: If template not found in either location
        """
        # Prefer Metal-specific templates
        base_dir = os.path.join(get_yapocis_dir(), "kernels", "metal")
        metal_template = os.path.join(base_dir, f"{base_filename}.mako")
        if os.path.exists(metal_template):
            return metal_template

        # Fallback to OCL kernels for compatibility
        ocl_dir = os.path.join(get_yapocis_dir(), "kernels", "ocl")
        ocl_template = os.path.join(ocl_dir, f"{base_filename}.mako")
        if os.path.exists(ocl_template):
            return ocl_template

        raise FileNotFoundError(
            f"No template found for {base_filename}; "
            f"looked in {base_dir} and {ocl_dir}"
        )

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
