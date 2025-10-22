import os

_backends = None
from . import get_backends, get_backend

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
    elif backend in ("numba","numbax"):
        # Return the Numba kernel_manager module (uses Numba JIT)
        from yapocis.numbax import kernel_manager as km

        return km
    raise ValueError(f"Unknown backend: {backend}")


def get_device(backend: str | None):
    manager = get_manager(backend)
    return manager.get_device()


def load_program(interface: str, backend: str | None = None, **mako_vars):
    """Backend-agnostic program loader. `interface` is the interface spec/name.

    Note: Parameter named `interface` avoids clashing with common Mako variable `name`.
    """
    mgr = get_manager(backend)
    return mgr.load_program(interface, **mako_vars)


def import_program(interface: str, backend: str | None = None, **mako_vars):
    mgr = get_manager(backend)
    return mgr.import_program(interface, **mako_vars)
