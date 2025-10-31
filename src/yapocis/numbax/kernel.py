from __future__ import annotations

from typing import Any, Callable

import numpy as np

class Kernel:
    """Callable wrapper for a Python/Numba kernel.

    Maps interface parameters to Python call arguments, allocating outputs
    for out_like and handling derived dims.
    """

    def __init__(self, name: str, func, params: list[tuple], kernel_manager=None, async_write: bool = False, traits: dict | None = None):
        self.name = name
        self.func = func
        self.params = params  # list of KernelParamSpec-like tuples/objects
        self.kernel_manager = kernel_manager
        self.async_write = async_write
        self.traits = traits or {}
        # Normalize to objects with attributes used below
        self._norm = []
        for p in params:
            # p may be a dataclass KernelParamSpec (core.interface.KernelParamSpec)
            self._norm.append(p)

    def write(self, a: Any, wait: bool = True):
        # No-op for in-memory arrays
        return a

    def read(self, a: Any):
        # No-op for in-memory arrays
        return a

    def __call__(self, *args: tuple[Any], **kwargs):
        # Consume async_exec parameter for API compatibility (no-op for CPU)
        # Honor kernel_manager's default async mode if available
        default_async = self.kernel_manager._default_async if self.kernel_manager else False
        async_exec = kwargs.pop("async_exec", default_async)

        prepared_args: list[Any] = [None] * len(self._norm)
        returns: list[np.ndarray] = []
        # Build map from logical arg name to user-supplied arg position
        posbyname = {}
        iarg = 0
        # First pass find non-derived argument positions
        for ip, p in enumerate(self._norm):
            if not p.derived:
                posbyname[p.name] = iarg
                iarg += 1
        # Prepare arguments
        for ip, p in enumerate(self._norm):
            if p.dim:
                base = (
                    args[posbyname[p.name]]
                    if p.name in posbyname
                    else args[posbyname[p.derived_from]]
                )
                if p.dim in ("size_of", "sizeof"):
                    prepared_args[ip] = int(np.prod(base.shape))
                elif p.dim in ("width_of", "widthof"):
                    prepared_args[ip] = int(base.shape[0])
                elif p.dim in ("height_of", "heightof"):
                    prepared_args[ip] = int(base.shape[1]) if base.ndim > 1 else 1
                elif p.dim in ("depth_of",):
                    prepared_args[ip] = int(base.shape[2]) if base.ndim > 2 else 1
                elif p.dim in ("length_of", "lengthof"):
                    prepared_args[ip] = int(max(base.shape))
                elif p.dim in ("count_of", "countof"):
                    # Number of elements (first dimension) for both 1D and 2D+
                    prepared_args[ip] = int(base.shape[0])
                else:
                    prepared_args[ip] = 0
            elif p.hint == "out_like":
                like = args[posbyname[p.derived_from]]
                # Allocate like-shaped output, but honor declared dtype
                dtype = getattr(p, "np_dtype", None) or getattr(p, "data_type", None)
                np_dtype = np.dtype(dtype) if dtype is not None else like.dtype
                out = np.empty(like.shape, dtype=np_dtype)
                prepared_args[ip] = out
                returns.append(out)
            elif p.hint == "out":
                out = args[posbyname[p.name]]
                prepared_args[ip] = out
                returns.append(out)
            else:
                # in / inout / resident or scalar
                if p.name in posbyname:
                    prepared_args[ip] = args[posbyname[p.name]]
                else:
                    # Derived scalar with no explicit position
                    prepared_args[ip] = None
        # Call underlying function
        rv = self.func(*prepared_args)

        # In async mode, return None for API consistency (even though execution is synchronous)
        if async_exec:
            return None

        if returns:
            return returns[0] if len(returns) == 1 else tuple(returns)
        return rv
