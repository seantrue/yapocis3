from __future__ import annotations
import cupy as cp
from pathlib import Path
from typing import Any
from core.interface import load_interface, get_interface
from yapocis.core.mako_utils import render_template, resolve_template
from yapocis.core.autopad import pad_sizes
from yapocis.core.tunecache import get_local_size
from yapocis.core.devbuf import ensure_device, empty_like_device
class CUDAProgram:
    def __init__(self, name: str, raw_module: cp.RawModule, iface, stream: cp.cuda.Stream, default_local=(16,16)):
        self.name=name; self.mod=raw_module; self.iface=iface; self.stream=stream; self.default_local=default_local
    def _grid_block(self, gs, ls):
        def cd(a,b): return (a + b - 1)//b
        if isinstance(gs, int):
            tx = (ls or (256,))[0]; return (cd(gs,tx),1,1),(tx,1,1)
        dims = list(gs or (1,)) + [1,1]; dims=dims[:3]
        lsz  = list(ls or (16,16,1)) + [1,1]; lsz=lsz[:3]
        grid = tuple(cd(dims[i], lsz[i]) for i in range(3))
        block= tuple(lsz); return grid, block
    def __getattr__(self, kname: str):
        cached_ls = get_local_size(self.name, kname, backend='cuda')
        kern = self.mod.get_function(kname)
        kspec = next((k for k in self.iface.kernels if k.name == kname), None)
        aspecs = getattr(kspec, "args", None)
        def call(*args: Any, global_size=None, local_size=None, **kwargs: Any):
            dev_args = list(args); out_idxs=[]
            if aspecs is not None and len(aspecs)==len(args):
                dev_args=[]; 
                for i,(aval,spec) in enumerate(zip(args, aspecs)):
                    q = getattr(spec, "qualifier", "in")
                    if q == "in":
                        dev_args.append(ensure_device(aval))
                    elif q in ("out","inout"):
                        if aval is None:
                            like_name = getattr(spec, "like", None); ref=None
                            if like_name is not None:
                                try:
                                    ref_idx = next(j for j,s in enumerate(aspecs) if s.name == like_name)
                                    ref = args[ref_idx]
                                except StopIteration:
                                    ref=None
                            dev = empty_like_device(ref) if ref is not None else ensure_device(aval)
                            dev_args.append(dev)
                        else:
                            dev_args.append(ensure_device(aval))
                        out_idxs.append(i)
                    else:
                        dev_args.append(ensure_device(aval))
            else:
                dev_args = [ensure_device(a) for a in args]
            preferred = cached_ls or self.default_local
            pg, pl = pad_sizes(global_size, local_size, tuple(args), preferred)
            gs = pg if pg is not None else global_size
            ls = pl if pl is not None else local_size
            grid, block = self._grid_block(gs, ls)
            kern(grid, block, tuple(dev_args), stream=self.stream)
            outs = [dev_args[i] for i in out_idxs] if out_idxs else []
            if not outs: return None
            return outs[0] if len(outs)==1 else tuple(outs)
        return call
class CUDAKernelManager:
    def __init__(self, templates_dir: str | Path | None = None, default_local=(16,16)):
        self.templates_dir = Path(templates_dir or Path(__file__).resolve().parents[2] / "cukernels")
        self.stream = cp.cuda.Stream(non_blocking=True); self.default_local=default_local
    def load_program(self, name: str, **mako_vars) -> CUDAProgram:
        tpath = resolve_template(self.templates_dir, name, backend="cuda")
        code = render_template(tpath, **mako_vars)
        mod = cp.RawModule(code=code, options=("--std=c++14",))
        iface_txt = load_interface(name); iface = get_interface(iface_txt)
        return CUDAProgram(name, mod, iface, self.stream, self.default_local)
    def import_program(self, name: str, namespace: bool = False, **mako_vars):
        prog = self.load_program(name, **mako_vars)
        callables = { k.name: getattr(prog, k.name) for k in prog.iface.kernels }
        if namespace:
            import inspect, sys
            mod = sys.modules[inspect.getmodule(self.import_program).__name__]
            for n, fn in callables.items(): setattr(mod, n, fn)
        return prog, callables
