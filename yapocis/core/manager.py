import os
def get_manager(backend: str | None = None):
    backend = (backend or os.environ.get("YAPOCIS_BACKEND") or "ocl").lower()
    if backend == "cuda":
        from yapocis.cu.kernel_manager import CUDAKernelManager
        return CUDAKernelManager()
    elif backend == "ocl":
        from yapocis.ocl.kernel_manager import KernelManager
        return KernelManager()
    raise ValueError(f"Unknown backend: {backend}")
def load_program(name: str, backend: str | None = None, **mako_vars):
    mgr = get_manager(backend); return mgr.load_program(name, **mako_vars)
def import_program(name: str, backend: str | None = None, **mako_vars):
    mgr = get_manager(backend); return mgr.import_program(name, **mako_vars)
