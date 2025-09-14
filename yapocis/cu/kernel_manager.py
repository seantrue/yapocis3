from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import pycuda.autoinit  # noqa: F401  (initializes CUDA context)
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    import pycuda.gpuarray as gpuarray
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyCUDA is required for the CUDA backend") from e

from yapocis.core.interface import load_interface, get_interface
from yapocis.core.mako_utils import render_template, resolve_template
from yapocis.core.autopad import pad_sizes
from yapocis.core.tunecache import get_local_size


def _to_device(x: Any):
    if isinstance(x, gpuarray.GPUArray):
        return x
    if isinstance(x, np.ndarray):
        return gpuarray.to_gpu(x)
    # Scalars: pass as-is (NumPy scalar for dtype)
    return np.array(x)


def _empty_like_device(ref: Any, dtype=None):
    if isinstance(ref, gpuarray.GPUArray):
        return gpuarray.empty_like(ref).astype(dtype or ref.dtype)
    if isinstance(ref, np.ndarray):
        shape = ref.shape; dt = dtype or ref.dtype
        return gpuarray.empty(shape, dt)
    raise TypeError("Cannot infer device buffer shape from reference")


class CUDAProgram:
    def __init__(self, name: str, module: SourceModule, iface, default_local=(16, 16)):
        self.name = name
        self.mod = module
        self.iface = iface
        self.default_local = default_local

    def _grid_block(self, gs, ls):
        def cd(a, b):
            return (a + b - 1) // b
        if gs is None:
            return (1, 1, 1), (1, 1, 1)
        if isinstance(gs, int):
            tx = (ls or (256,))[0]
            return (cd(gs, tx), 1, 1), (tx, 1, 1)
        dims = list(gs or (1,)) + [1, 1]
        dims = dims[:3]
        lsz = list(ls or (16, 16, 1)) + [1, 1]
        lsz = lsz[:3]
        grid = tuple(cd(dims[i], lsz[i]) for i in range(3))
        block = tuple(lsz)
        return grid, block

    def __getattr__(self, kname: str):
        cached_ls = get_local_size(self.name, kname, backend="cuda")
        func = self.mod.get_function(kname)
        kspec = next((k for k in self.iface.kernels if k.name == kname), None)
        aspecs = getattr(kspec, "args", None)

        def call(*args: Any, global_size=None, local_size=None, **kwargs: Any):
            dev_args = []
            out_idxs = []
            if aspecs is not None and len(aspecs) == len(args):
                for i, (aval, spec) in enumerate(zip(args, aspecs)):
                    q = getattr(spec, "qualifier", "in")
                    if q == "in":
                        dev_args.append(_to_device(aval))
                    elif q in ("out", "inout"):
                        if aval is None:
                            like_name = getattr(spec, "like", None)
                            ref = None
                            if like_name is not None:
                                try:
                                    ref_idx = next(j for j, s in enumerate(aspecs) if s.name == like_name)
                                    ref = args[ref_idx]
                                except StopIteration:
                                    ref = None
                            dev = _empty_like_device(_to_device(ref)) if ref is not None else _to_device(aval)
                            dev_args.append(dev)
                        else:
                            dev_args.append(_to_device(aval))
                        out_idxs.append(i)
                    else:
                        dev_args.append(_to_device(aval))
            else:
                dev_args = [_to_device(a) for a in args]

            preferred = cached_ls or self.default_local
            pg, pl = pad_sizes(global_size, local_size, tuple(args), preferred)
            gs = pg if pg is not None else global_size
            ls = pl if pl is not None else local_size
            grid, block = self._grid_block(gs, ls)

            # Unwrap GPUArrays to device pointers or pass scalars
            launch_args = [a.gpudata if isinstance(a, gpuarray.GPUArray) else a for a in dev_args]
            func(*launch_args, block=block, grid=grid)

            outs = [dev_args[i] for i in out_idxs] if out_idxs else []
            if not outs:
                return None
            # Return host arrays for convenience (match OCL/Metal behavior)
            host_outs = [o.get() if isinstance(o, gpuarray.GPUArray) else o for o in outs]
            return host_outs[0] if len(host_outs) == 1 else tuple(host_outs)

        return call


class CUDAKernelManager:
    def __init__(self, templates_dir: str | Path | None = None, default_local=(16, 16)):
        self.templates_dir = Path(templates_dir or Path(__file__).resolve().parents[2] / "cukernels")
        self.default_local = default_local

    def load_program(self, name: str, **mako_vars) -> CUDAProgram:
        tpath = resolve_template(self.templates_dir, name, backend="cuda")
        code = render_template(tpath, **mako_vars)
        # Compile with NVCC via PyCUDA
        mod = SourceModule(code, options=["-std=c++11"])  # adjust if templates use C++
        iface_txt = load_interface(name)
        iface = get_interface(iface_txt)
        prog = CUDAProgram(name, mod, iface, self.default_local)
        return prog

    def import_program(self, name: str, namespace: bool = False, **mako_vars):
        prog = self.load_program(name, **mako_vars)
        callables = {k.name: getattr(prog, k.name) for k in prog.iface.kernels}
        if namespace:
            import inspect, sys
            mod = sys.modules[inspect.getmodule(self.import_program).__name__]
            for n, fn in callables.items():
                setattr(mod, n, fn)
        return prog, callables
