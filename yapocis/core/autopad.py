from __future__ import annotations
from typing import Any, Optional, Sequence, Tuple

try:
    import numpy as _np
except Exception:
    _np = None


def _is_array(x: Any) -> bool:
    if _np is not None and isinstance(x, _np.ndarray):
        return True
    # Detect PyCUDA GPUArray without importing at module import time if unavailable
    try:
        import pycuda.gpuarray as _ga  # type: ignore
        if isinstance(x, _ga.GPUArray):
            return True
    except Exception:
        pass
    return False


def infer_wh_from_args(args: Tuple[Any, ...]) -> Optional[Tuple[int, int]]:
    for a in args:
        if _is_array(a):
            return a.shape
    return None


def _ceil_div(a: int, b: int) -> int: return (a + b - 1) // b if b else a


def pad_sizes(global_size: Optional[Sequence[int]], local_size: Optional[Sequence[int]], args: Tuple[Any, ...],
              default_local: Optional[Sequence[int]] = None, tiled: bool=False):
    gs = None
    if global_size is None:
        wh = infer_wh_from_args(args)
        if wh is not None:
            gs = wh
    else:
        gs = tuple(int(x) for x in global_size)
    if gs is None:
        return None, tuple(default_local) if default_local is not None else None
    ls = tuple(int(x) for x in (local_size or default_local or ())) or None
    if ls is None: return tuple(gs), None
    g = list(gs);
    l = list(ls)
    while len(g) < len(l): g.append(1)
    while len(l) < len(g): l.append(1)
    pg = tuple(_ceil_div(g[i], l[i]) * l[i] for i in range(len(g)))
    return pg, tuple(l)
# def wrap_callable(fn, *, default_local=None):
#     def wrapped(*args, global_size=None, local_size=None, **kwargs):
#         pg, pl = pad_sizes(global_size, local_size, args, default_local)
#         if pg is None: return fn(*args, global_size=global_size, local_size=local_size, **kwargs)
#         return fn(*args, global_size=pg, local_size=pl or local_size, **kwargs)
#     wrapped.__wrapped_kernel__ = fn
#     return wrapped
