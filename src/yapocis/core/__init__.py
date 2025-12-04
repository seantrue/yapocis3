from __future__ import annotations

import os
from typing import Any

import numpy as np


def infer_wh_from_args(args: tuple[Any, ...]) -> tuple[int, int] | None:
    for a in args:
        if isinstance(a, np.ndarray):
            return a.shape
    return None


_backends = None

ALL_BACKENDS = ("ocl", "metal", "cuda", "numbax")


def get_all_backends():
    return ALL_BACKENDS


def get_backends() -> list[str]:
    global _backends
    if _backends is None:
        backends = []
        try:
            import pyopencl

            backends.append("ocl")
        except:
            pass
        try:
            import pymetallic

            backends.append("metal")
        except:
            pass
        try:
            import pycuda

            backends.append("cuda")
        except:
            pass
        try:
            import numba

            backends.append("numbax")
        except:
            pass
        _backends = backends
    return _backends


def get_backend():
    # Returns active backend
    backend = os.environ.get("YAPOCIS_BACKEND") or "ocl"
    backend = backend.lower()
    backends = get_backends()
    if backend not in backends:
        raise RuntimeError(f"{backend=} is not one of available {backends=}")
    return backend


def get_backend_device():
    # Returns device for active backend
    backend = get_backend()
    info = get_backend_info(backend)
    devices = info["devices"]
    selected_device = os.environ.get("YAPOCIS_DEVICE", None)
    if selected_device is None:
        return (0, devices[0])

    for i, name in enumerate(info["devices"]):
        if selected_device is not None:
            # Check if device matches by index
            if selected_device.isdigit() and int(selected_device) == i:
                return (i, devices[i])
            # Or if the device name matches exactly
            elif not selected_device.isdigit() and selected_device == devices[i]:
                return (i, devices[i])
    return (0, devices[0])


def get_manager(backend: str | None = None):
    """
    Return a backend manager module exposing `load_program` and `import_program`.

    Historically this returned instantiated KernelManager objects, but the
    constructors require an interface and device. Returning the module keeps
    the API simple and consistent: callers use `mgr.load_program(...)`.
    """
    backend = get_backend()
    if backend == "cuda":
        # Return the CUDA kernel_manager module (has load_program/import_program)
        from yapocis.cu import kernel_manager as km

        return km
    elif backend in ("ocl", "opencl"):
        # Return the OpenCL kernel_manager module (has load_program/import_program)
        from yapocis.ocl import kernel_manager as km

        return km
    elif backend in ("metal", "metallic", "mtl"):
        # Return the Metal kernel_manager module (uses pymetallic runtime)
        from yapocis.metallic import kernel_manager as km

        return km
    elif backend in ("numbax", "numba"):
        # Return the Numba kernel_manager module (uses Numba JIT)
        from yapocis.numbax import kernel_manager as km

        return km
    raise ValueError(f"Unknown backend: {backend}")


def load_program(interface: str, backend: str | None = None, **mako_vars):
    """Backend-agnostic program loader. `interface` is the interface spec/name.

    Note: Parameter named `interface` avoids clashing with common Mako variable `name`.
    """
    mgr = get_manager(backend)
    return mgr.load_program(interface, **mako_vars)


def import_program(interface: str, backend: str | None = None, **mako_vars):
    mgr = get_manager(backend)
    return mgr.import_program(interface, **mako_vars)


_backend_info = None


def get_backend_infos() -> dict[str, Any]:
    """Get information about all available backends and their devices."""
    global _backend_info
    if _backend_info is None:
        backend_info = {}

        for backend in get_all_backends():
            backend_info[backend] = {"available": True, "devices": [], "error": None}

            try:
                if backend == "ocl":
                    from yapocis.ocl.device import get_available_devices

                    devices = get_available_devices(aliases=False)
                    backend_info[backend]["devices"] = list(devices.keys())

                elif backend == "metal":
                    from yapocis.metallic.device import get_available_devices

                    devices = get_available_devices(aliases=False)
                    backend_info[backend]["devices"] = list(devices.keys())

                elif backend == "cuda":
                    try:
                        from yapocis.cu.device import get_available_devices

                        devices = get_available_devices()
                        # Filter out numeric keys (indices) to get only device names
                        backend_info[backend]["devices"] = [
                            name for name in devices.keys() if isinstance(name, str)
                        ]
                    except Exception as e:
                        backend_info[backend]["error"] = str(e)
                        backend_info[backend]["available"] = False

                elif backend == "numbax":
                    # Numba runs on CPU, no separate device enumeration
                    import platform

                    backend_info[backend]["devices"] = [
                        f"CPU: {platform.processor() or platform.machine()}"
                    ]

            except Exception as e:
                backend_info[backend]["error"] = str(e)
                backend_info[backend]["available"] = False
        _backend_info = backend_info
    return _backend_info


def get_backend_info(backend):
    return get_backend_infos()[backend]


def get_active_backend():
    backend = get_backend()
    backend_info = get_backend_info(backend)


def get_available_devices(backend):
    info = get_backend_info(backend)
    return info["devices"]


def matches_filter(name: str, filters: list[str]) -> bool:
    """
    Check if name matches any of the filters (substring match).

    Args:
        name: Name to check
        filters: List of filter strings (case-insensitive substring match)

    Returns:
        True if name matches any filter, or if filters is empty
    """
    if not filters:
        return True

    name_lower = name.lower()
    return any(f.lower() in name_lower for f in filters)


def get_test_backends(backend_filters: list[str] | None = None) -> list[str]:
    """
    Get list of backends to test, applying filters and environment variables.

    Priority:
    1. If backend_filters is explicitly provided (not None), use those
    2. Else check YAPOCIS_TEST_BACKENDS environment variable
    3. Else return all available backends

    Args:
        backend_filters: Optional list of backend names to filter

    Returns:
        List of backend names to test

    Environment:
        YAPOCIS_TEST_BACKENDS: Comma-separated list of backends (e.g., "ocl,cuda")
    """
    # Get all available backends
    available_backends = get_backends()

    # Determine which filters to use
    if backend_filters is None:
        # Check environment variable
        env_backends = os.environ.get("YAPOCIS_TEST_BACKENDS", "").strip()
        if env_backends:
            backend_filters = [b.strip() for b in env_backends.split(",") if b.strip()]
        else:
            backend_filters = []

    # Apply filters
    if backend_filters:
        available_backends = [
            b for b in available_backends if matches_filter(b, backend_filters)
        ]

    return available_backends


def get_test_devices(
    backend: str, device_filters: list[str] | None = None
) -> dict[str, str]:
    """
    Get devices for a backend, applying filters and environment variables.

    Priority:
    1. If device_filters is explicitly provided (not None), use those
    2. Else check YAPOCIS_TEST_DEVICES environment variable
    3. Else return all available devices

    Device filters are only applied to GPU backends (ocl, metal, cuda).
    CPU-only backends (numbax) always return their single device.

    Args:
        backend: Backend name (ocl, metal, cuda, numbax)
        device_filters: Optional list of device name filters

    Returns:
        Dict mapping device name to full identifier (backend:device)

    Environment:
        YAPOCIS_TEST_DEVICES: Comma-separated list of device filters (e.g., "GTX,Quadro")
    """
    # Determine which filters to use
    if device_filters is None:
        # Check environment variable
        env_devices = os.environ.get("YAPOCIS_TEST_DEVICES", "").strip()
        if env_devices:
            device_filters = [d.strip() for d in env_devices.split(",") if d.strip()]
        else:
            device_filters = []

    devices = {}

    try:
        if backend == "ocl":
            from yapocis.ocl.device import get_available_devices

            dev_dict = get_available_devices(aliases=False)
            for name in dev_dict.keys():
                if isinstance(name, str) and matches_filter(name, device_filters):
                    devices[name] = f"ocl:{name}"

        elif backend == "metal":
            from yapocis.metallic.device import get_available_devices

            dev_dict = get_available_devices(aliases=False)
            for name in dev_dict.keys():
                if isinstance(name, str) and matches_filter(name, device_filters):
                    devices[name] = f"metal:{name}"

        elif backend == "cuda":
            from yapocis.cu.device import get_available_devices

            dev_dict = get_available_devices(aliases=False)
            for name in dev_dict.keys():
                if isinstance(name, str) and matches_filter(name, device_filters):
                    devices[name] = f"cuda:{name}"

        elif backend == "numbax":
            # Numba is CPU-based, single "device"
            # Don't apply device filters to CPU-only backends
            devices["CPU (Numba JIT)"] = "numbax:cpu"

    except Exception:
        # Silently skip backends that fail to enumerate devices
        pass

    return devices
