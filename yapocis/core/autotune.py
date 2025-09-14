
"""
Auto-tuning helpers for choosing local_size / tile sizes.
- CUDA: uses PyCUDA events for accurate timing when available.
- Fallback: uses time.perf_counter with best-effort synchronization.
"""
from __future__ import annotations
from typing import Iterable, Sequence, Tuple, Any, Dict, Optional
import time

try:
    import pycuda.driver as _cuda  # type: ignore
except Exception:  # pragma: no cover
    _cuda = None

def _sync_cuda():
    if _cuda is not None:
        try:
            _cuda.Context.synchronize()
        except Exception:
            pass

def _time_kernel_cuda(fn, args, *, global_size, local_size, iters=10, warmup=2) -> float:
    if _cuda is None:
        raise RuntimeError("PyCUDA not available")
    start = _cuda.Event()
    end = _cuda.Event()
    # warmup
    for _ in range(warmup):
        fn(*args, global_size=global_size, local_size=local_size)
    _sync_cuda()
    start.record()
    for _ in range(iters):
        fn(*args, global_size=global_size, local_size=local_size)
    end.record()
    end.synchronize()
    try:
        # Prefer end.time_since(start) if available
        elapsed = end.time_since(start)
    except Exception:
        elapsed = start.time_till(end)
    return elapsed / iters  # milliseconds

def _time_kernel_fallback(fn, args, *, global_size, local_size, iters=10, warmup=2) -> float:
    for _ in range(warmup):
        fn(*args, global_size=global_size, local_size=local_size)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args, global_size=global_size, local_size=local_size)
    _sync_cuda()  # in case we're on CUDA but events unavailable
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters  # ms

def autotune_local_size(fn, args: Sequence[Any], *, global_size: Sequence[int],
                        candidates: Iterable[Sequence[int]],
                        iters: int = 20, warmup: int = 3,
                        prefer_cuda_timer: bool = True) -> Dict[str, Any]:
    """
    Try a list of local_size candidates, pick the fastest.
    Returns: {"best": (lx,ly[,lz]), "ms": best_ms, "results": [(ls, ms), ...]}
    """
    results = []
    timer = None
    if prefer_cuda_timer and _cuda is not None:
        timer = _time_kernel_cuda
    if timer is None:
        timer = _time_kernel_fallback

    best = None
    best_ms = float("inf")
    for ls in candidates:
        ls_t = tuple(int(x) for x in ls)
        try:
            ms = timer(fn, args, global_size=tuple(global_size), local_size=ls_t, iters=iters, warmup=warmup)
            results.append((ls_t, ms))
            if ms < best_ms:
                best_ms = ms
                best = ls_t
        except Exception as e:  # invalid configuration, skip
            results.append((ls_t, None))
            continue
    return {"best": best, "ms": best_ms, "results": results}


from .tunecache import set_local_size
def autotune_and_cache(program_name: str, kernel_name: str, fn, args, *, global_size, candidates, backend="cuda", tag=None, iters=20, warmup=3):
    res = autotune_local_size(fn, args, global_size=global_size, candidates=candidates, iters=iters, warmup=warmup)
    if res.get("best"):
        set_local_size(program_name, kernel_name, res["best"], backend=backend, tag=tag)
    return res
