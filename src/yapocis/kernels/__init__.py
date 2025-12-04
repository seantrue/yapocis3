"""Kernel templates organized by backend."""

#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.

import glob
import os

from yapocis.utils.internals import get_yapocis_dir


def kernels():
    """Return list of all kernel templates across all backends."""
    kernel_names = []
    kernels_dir = os.path.join(get_yapocis_dir(), "kernels")

    # Scan all backend subdirectories
    for backend in ["ocl", "metal", "cuda", "numbax"]:
        backend_dir = os.path.join(kernels_dir, backend)
        if os.path.isdir(backend_dir):
            # Find all .mako files (GPU backends) or .py files (numbax)
            for ext in [".mako", ".cl", ".metal", ".cu", ".py"]:
                pattern = os.path.join(backend_dir, f"*{ext}")
                for path in glob.glob(pattern):
                    basename = os.path.basename(path)
                    # Remove extension to get kernel name
                    name = os.path.splitext(basename)[0]
                    if name and name not in kernel_names and not name.startswith("__"):
                        kernel_names.append(name)

    return sorted(kernel_names)


def find_kernel(name):
    """Find kernel template file by name across all backends.

    Raises ValueError if not found.
    """
    kernels_dir = os.path.join(get_yapocis_dir(), "kernels")

    # Search in all backend directories
    for backend in ["ocl", "metal", "cuda", "numbax"]:
        backend_dir = os.path.join(kernels_dir, backend)
        if os.path.isdir(backend_dir):
            # Try different extensions
            for ext in [".mako", ".cl", ".metal", ".cu", ".py"]:
                candidate = os.path.join(backend_dir, f"{name}{ext}")
                if os.path.exists(candidate):
                    return candidate

    raise ValueError(f"Kernel '{name}' not found in any backend directory")
