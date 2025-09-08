from __future__ import annotations
from typing import Tuple, Any
import numpy as np
try:
    import cupy as cp
except Exception:
    cp = None
def ensure_device(x: Any):
    if cp is not None and isinstance(x, cp.ndarray):
        return x
    if isinstance(x, np.ndarray):
        return cp.asarray(x) if cp is not None else x
    return x
def empty_like_device(ref: Any, dtype=None):
    if cp is not None and isinstance(ref, cp.ndarray):
        return cp.empty_like(ref, dtype=dtype or ref.dtype)
    if isinstance(ref, np.ndarray) and cp is not None:
        return cp.empty_like(cp.asarray(ref), dtype=dtype or ref.dtype)
    return np.empty_like(ref, dtype=dtype or ref.dtype)
def empty_device(shape: Tuple[int, ...], dtype):
    if cp is not None:
        return cp.empty(shape, dtype=dtype)
    return np.empty(shape, dtype=dtype)
def to_host(x: Any):
    if cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return x
