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
from yapocis.utils.internals import get_yapocis_dir, get_yapocis_dirs

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

    def find_kernel_template(self, base_filename: str) -> str:
        """
        Metal-specific template resolution with OCL fallback.

        Searches in order across all yapocis directories (YAPOCIS_PROJECT, cwd, package):
        1. kernels/metal/*.mako (Metal-specific)
        2. kernels/ocl/*.mako (OCL fallback)

        Args:
            base_filename: Interface name (e.g., "mandelbrot")

        Returns:
            Full path to template file

        Raises:
            FileNotFoundError: If template not found in any location
        """
        template_filename = f"{base_filename}.mako"
        searched_paths = []

        # Search all yapocis directories for Metal-specific template first
        for base_dir in get_yapocis_dirs():
            metal_template = os.path.join(base_dir, "kernels", "metal", template_filename)
            searched_paths.append(metal_template)
            if os.path.exists(metal_template):
                return metal_template

        # Fallback: search all directories for OCL template
        for base_dir in get_yapocis_dirs():
            ocl_template = os.path.join(base_dir, "kernels", "ocl", template_filename)
            searched_paths.append(ocl_template)
            if os.path.exists(ocl_template):
                return ocl_template

        raise FileNotFoundError(
            f"No template found for '{base_filename}' (backend=metal). "
            f"Searched: {', '.join(searched_paths)}"
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
