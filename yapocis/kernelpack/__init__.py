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
    from yapocis.kernelpack import get_convolvesep_package

    package = get_convolvesep_package(dmgr)
    result = package.gauss('gauss3', vertical=0, input_array=image)
"""

from yapocis.core.manager import get_backend


def get_convolvesep_package(kmgr):
    """
    Get the compiled ConvolvesepPackage for current backend.

    Args:
        kmgr: Kernel manager (has dmgr and program attributes)

    Returns:
        ConvolvesepPackage instance for the appropriate backend

    Raises:
        ImportError: If package not generated for this backend
        ValueError: If backend is unknown
    """
    backend = get_backend()

    try:
        if backend == "ocl":
            from yapocis.kernelpack.ocl.convolvesep import ConvolvesepPackage
        elif backend == "metal":
            from yapocis.kernelpack.metal.convolvesep import ConvolvesepPackage
        elif backend == "cuda":
            from yapocis.kernelpack.cuda.convolvesep import ConvolvesepPackage
        elif backend == "numbax":
            from yapocis.kernelpack.numbax.convolvesep import ConvolvesepPackage
        else:
            raise ValueError(f"Unknown backend: {backend}")
    except ImportError as e:
        raise ImportError(
            f"Convolvesep package not generated for backend '{backend}'. "
            f"Run: yapocis-package convolvesep --backend {backend}"
        ) from e

    return ConvolvesepPackage(kmgr.dmgr, kmgr.program)


def get_operators_package(kmgr):
    """
    Get the compiled OperatorsPackage for current backend.

    Args:
        kmgr: Kernel manager (has dmgr and program attributes)

    Returns:
        OperatorsPackage instance for the appropriate backend

    Raises:
        ImportError: If package not generated for this backend
        ValueError: If backend is unknown
    """
    backend = get_backend()

    try:
        if backend == "ocl":
            from yapocis.kernelpack.ocl.operators import OperatorsPackage
        elif backend == "metal":
            from yapocis.kernelpack.metal.operators import OperatorsPackage
        elif backend == "cuda":
            from yapocis.kernelpack.cuda.operators import OperatorsPackage
        elif backend == "numbax":
            from yapocis.kernelpack.numbax.operators import OperatorsPackage
        else:
            raise ValueError(f"Unknown backend: {backend}")
    except ImportError as e:
        raise ImportError(
            f"Operators package not generated for backend '{backend}'. "
            f"Run: yapocis-package operators --backend {backend}"
        ) from e

    return OperatorsPackage(kmgr.dmgr, kmgr.program)


def get_gradient_package(kmgr):
    """Get the compiled GradientPackage for current backend."""
    backend = get_backend()

    try:
        if backend == "ocl":
            from yapocis.kernelpack.ocl.gradient import GradientPackage
        elif backend == "metal":
            from yapocis.kernelpack.metal.gradient import GradientPackage
        elif backend == "cuda":
            from yapocis.kernelpack.cuda.gradient import GradientPackage
        elif backend == "numbax":
            from yapocis.kernelpack.numbax.gradient import GradientPackage
        else:
            raise ValueError(f"Unknown backend: {backend}")
    except ImportError as e:
        raise ImportError(
            f"Gradient package not generated for backend '{backend}'. "
            f"Run: yapocis-package gradient --backend {backend}"
        ) from e

    return GradientPackage(kmgr.dmgr, kmgr.program)


def get_median3x3_package(kmgr):
    """Get the compiled Median3x3Package for current backend."""
    backend = get_backend()
    try:
        if backend == "ocl":
            from yapocis.kernelpack.ocl.median3x3 import Median3x3Package
        elif backend == "metal":
            from yapocis.kernelpack.metal.median3x3 import Median3x3Package
        elif backend == "cuda":
            from yapocis.kernelpack.cuda.median3x3 import Median3x3Package
        elif backend == "numbax":
            from yapocis.kernelpack.numbax.median3x3 import Median3x3Package
        else:
            raise ValueError(f"Unknown backend: {backend}")
    except ImportError as e:
        raise ImportError(f"Median3x3 package not generated for backend '{backend}'.") from e
    return Median3x3Package(kmgr.dmgr, kmgr.program)


def get_colormath_package(kmgr):
    """Get the compiled ColormathPackage for current backend."""
    backend = get_backend()
    try:
        if backend == "ocl":
            from yapocis.kernelpack.ocl.colormath import ColormathPackage
        elif backend == "metal":
            from yapocis.kernelpack.metal.colormath import ColormathPackage
        elif backend == "cuda":
            from yapocis.kernelpack.cuda.colormath import ColormathPackage
        elif backend == "numbax":
            from yapocis.kernelpack.numbax.colormath import ColormathPackage
        else:
            raise ValueError(f"Unknown backend: {backend}")
    except ImportError as e:
        raise ImportError(f"Colormath package not generated for backend '{backend}'.") from e
    return ColormathPackage(kmgr.dmgr, kmgr.program)


def get_hsi_package(kmgr):
    """Get the compiled HsiPackage for current backend."""
    backend = get_backend()
    try:
        if backend == "ocl":
            from yapocis.kernelpack.ocl.hsi import HsiPackage
        elif backend == "metal":
            from yapocis.kernelpack.metal.hsi import HsiPackage
        elif backend == "cuda":
            from yapocis.kernelpack.cuda.hsi import HsiPackage
        elif backend == "numbax":
            from yapocis.kernelpack.numbax.hsi import HsiPackage
        else:
            raise ValueError(f"Unknown backend: {backend}")
    except ImportError as e:
        raise ImportError(f"HSI package not generated for backend '{backend}'.") from e
    return HsiPackage(kmgr.dmgr, kmgr.program)


def get_zcs_package(kmgr):
    """Get the compiled ZcsPackage for current backend."""
    backend = get_backend()
    try:
        if backend == "ocl":
            from yapocis.kernelpack.ocl.zcs import ZcsPackage
        elif backend == "metal":
            from yapocis.kernelpack.metal.zcs import ZcsPackage
        elif backend == "cuda":
            from yapocis.kernelpack.cuda.zcs import ZcsPackage
        elif backend == "numbax":
            from yapocis.kernelpack.numbax.zcs import ZcsPackage
        else:
            raise ValueError(f"Unknown backend: {backend}")
    except ImportError as e:
        raise ImportError(f"ZCS package not generated for backend '{backend}'.") from e
    return ZcsPackage(kmgr.dmgr, kmgr.program)


def get_anisodiff_package(kmgr):
    """Get the compiled AnisodiffPackage for current backend."""
    backend = get_backend()
    try:
        if backend == "ocl":
            from yapocis.kernelpack.ocl.anisodiff import AnisodiffPackage
        elif backend == "metal":
            from yapocis.kernelpack.metal.anisodiff import AnisodiffPackage
        elif backend == "cuda":
            from yapocis.kernelpack.cuda.anisodiff import AnisodiffPackage
        elif backend == "numbax":
            from yapocis.kernelpack.numbax.anisodiff import AnisodiffPackage
        else:
            raise ValueError(f"Unknown backend: {backend}")
    except ImportError as e:
        raise ImportError(f"Anisodiff package not generated for backend '{backend}'.") from e
    return AnisodiffPackage(kmgr.dmgr, kmgr.program)


def get_graphics4_package(kmgr):
    """Get the compiled Graphics4Package for current backend."""
    backend = get_backend()
    try:
        if backend == "ocl":
            from yapocis.kernelpack.ocl.graphics4 import Graphics4Package
        elif backend == "metal":
            from yapocis.kernelpack.metal.graphics4 import Graphics4Package
        elif backend == "cuda":
            from yapocis.kernelpack.cuda.graphics4 import Graphics4Package
        elif backend == "numbax":
            from yapocis.kernelpack.numbax.graphics4 import Graphics4Package
        else:
            raise ValueError(f"Unknown backend: {backend}")
    except ImportError as e:
        raise ImportError(f"Graphics4 package not generated for backend '{backend}'.") from e
    return Graphics4Package(kmgr.dmgr, kmgr.program)


def get_truetrace_package(kmgr):
    """Get the compiled TruetracePackage for current backend."""
    backend = get_backend()
    try:
        if backend == "ocl":
            from yapocis.kernelpack.ocl.truetrace import TruetracePackage
        elif backend == "metal":
            from yapocis.kernelpack.metal.truetrace import TruetracePackage
        elif backend == "cuda":
            from yapocis.kernelpack.cuda.truetrace import TruetracePackage
        elif backend == "numbax":
            from yapocis.kernelpack.numbax.truetrace import TruetracePackage
        else:
            raise ValueError(f"Unknown backend: {backend}")
    except ImportError as e:
        raise ImportError(f"Truetrace package not generated for backend '{backend}'.") from e
    return TruetracePackage(kmgr.dmgr, kmgr.program)


__all__ = [
    "get_convolvesep_package",
    "get_operators_package",
    "get_gradient_package",
    "get_median3x3_package",
    "get_colormath_package",
    "get_hsi_package",
    "get_zcs_package",
    "get_anisodiff_package",
    "get_graphics4_package",
    "get_truetrace_package",
]
