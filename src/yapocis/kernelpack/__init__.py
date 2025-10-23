"""
Compiled kernel packages for zero-overhead GPU kernel calls.

This package contains pre-compiled Python classes generated from interface
specifications. These eliminate runtime interpretation overhead by
pre-compiling all buffer management logic into direct method calls.

Directory structure:
    ocl/      - OpenCL compiled packages
    metal/    - Metal compiled packages
    cuda/     - CUDA compiled packages
    numbax/    - Numba compiled packages

Usage:
    from yapocis.kernelpack import get_kernelpack

    package = get_kernelpack('median3x3', kernel_manager)
    package.load()
    result = package.median3x3(input_array)
"""

from typing import Optional
from pathlib import Path
from yapocis.core.manager import get_backend


def get_kernel_names(interface: str) -> list[str]:
    """
    Extract kernel names from interface specification.

    Args:
        interface: Interface name (e.g., 'median3x3', 'gradient')

    Returns:
        List of kernel names defined in the interface

    Example:
        >>> get_kernel_names('gradient')
        ['gradient']
        >>> get_kernel_names('graphics4')
        ['lines', 'rects', 'circles', 'triangles', 'ellipses', 'composite']
    """
    from yapocis.interfaces import find_interface
    from yapocis.core.interface_parser import parse_interface_text

    # Find and read interface file
    interface_path = Path(find_interface(interface))
    interface_text = interface_path.read_text()

    # Parse interface specification
    specs = parse_interface_text(interface_text)
    if not specs:
        return []

    # Extract kernel names (excluding extensions and aliases, just base kernels)
    spec = specs[0]
    kernel_names = []
    for kernel in spec.kernels:
        # Skip if this is an extension or alias
        if kernel.extends or kernel.alias_of:
            continue
        kernel_names.append(kernel.name)

    return kernel_names

# Mapping of interface names to package class names
PACKAGE_CLASS_NAMES = {
    'convolvesep': 'ConvolvesepPackage',
    'operators':   'OperatorsPackage',
    'gradient':    'GradientPackage',
    'median3x3':   'Median3x3Package',
    'colormath':   'ColormathPackage',
    'hsi':         'HsiPackage',
    'zcs':         'ZcsPackage',
    'anisodiff':   'AnisodiffPackage',
    'graphics4':   'Graphics4Package',
    'truetrace':   'TruetracePackage',
}


def get_kernelpack(interface: str, kernel_manager, backend: Optional[str] = None):
    """
    Dynamically load kernel package for given interface and backend.

    This unified API replaces the individual get_X_package() functions.
    It dynamically imports and instantiates the appropriate package class
    based on the interface name and backend.

    Args:
        interface: Interface name (e.g., 'median3x3', 'gradient', 'operators')
        kernel_manager: Kernel manager instance (has dmgr and program attributes)
        backend: Backend name ('ocl', 'metal', 'cuda', 'numbax').
                If None, uses current backend from environment.

    Returns:
        Package instance for the specified interface and backend.
        Note: Call package.load() before executing kernels.

    Raises:
        ValueError: If interface or backend is unknown
        ImportError: If package not generated for this backend

    Examples:
        >>> from yapocis.core import load_program
        >>> from yapocis.kernelpack import get_kernelpack
        >>>
        >>> # Auto-detect backend from environment
        >>> kmgr = load_program('median3x3')
        >>> package = get_kernelpack('median3x3', kmgr)
        >>> package.load()
        >>> result = package.median3x3(input_array)
        >>>
        >>> # Explicit backend specification
        >>> kmgr = load_program('gradient')
        >>> package = get_kernelpack('gradient', kmgr, backend='cuda')
        >>> package.load()
        >>> grad, theta = package.gradient(image, 3)
        >>>
        >>> # Multi-kernel interface with selective loading
        >>> kmgr = load_program('operators')
        >>> package = get_kernelpack('operators', kmgr)
        >>> package.load(['add', 'sub', 'mul'])
        >>> result = package.add(a, b)
    """
    # Validate interface name
    if interface not in PACKAGE_CLASS_NAMES:
        available = ', '.join(sorted(PACKAGE_CLASS_NAMES.keys()))
        raise ValueError(
            f"Unknown interface '{interface}'. "
            f"Available interfaces: {available}"
        )

    # Get backend (use provided or auto-detect)
    if backend is None:
        backend = get_backend()

    # Validate backend
    if backend not in ('ocl', 'metal', 'cuda', 'numbax'):
        raise ValueError(
            f"Unknown backend '{backend}'. "
            f"Valid backends: ocl, metal, cuda, numbax"
        )

    # Get package class name
    class_name = PACKAGE_CLASS_NAMES[interface]

    # Dynamically import package class
    module_path = f"yapocis.kernelpack.{backend}.{interface}"
    try:
        module = __import__(module_path, fromlist=[class_name])
        package_class = getattr(module, class_name)
    except ImportError as e:
        raise ImportError(
            f"{class_name} not generated for backend '{backend}'. "
            f"Run: yapocis-package {interface} --backend {backend}"
        ) from e
    except AttributeError as e:
        raise ImportError(
            f"Package module exists but {class_name} not found. "
            f"The package may be corrupted or incompletely generated."
        ) from e

    # Instantiate and return package
    return package_class(kernel_manager.dmgr, kernel_manager.program)


__all__ = ["get_kernelpack", "get_kernel_names", "PACKAGE_CLASS_NAMES"]
