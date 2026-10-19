#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.
"""
Derivation fixtures: what the Python runtime computes from user arguments.

For every kernel in a bundle and a few synthetic argument shapes, record the
kernel argument list the Python backends would build (derived dimension values,
``out_like`` allocations, argument order), the ``dispatch_size`` result and the
per-backend default global/local sizes. A non-Python runtime replays these to
prove it derives the same values, which is the first line of defence against the
two implementations drifting apart.

The values come from the real code paths where one exists without a device:
``KernelBase.handle_*`` for dimensions and ``yapocis.core.dispatch_expr`` for
dispatch sizes. The per-backend size defaults mirror ``_normalize_sizes`` in
``ocl/kernel.py``, ``metallic/kernel.py`` and ``cu/kernel.py``.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from yapocis.core.dispatch_expr import evaluate_dispatch_size
from yapocis.core.interface import (
    InterfaceSpec,
    KernelParamSpec,
    KernelSpec,
    get_interface,
)
from yapocis.core.kernel_base import KernelBase

from .bundle import Manifest, evaluate_dispatch_ast

# Synthetic shapes: name -> array shape used for every buffer argument
CASES: dict[str, tuple[int, ...]] = {
    "1d": (11,),
    "2d": (7, 5),
    "3d": (3, 7, 5),
}
SCALAR_VALUE = 3


class _Deriver(KernelBase):
    """KernelBase with the abstract backend hooks stubbed, to run the handle_* methods."""

    def _package_dimension(self, value, dtype):
        return int(dtype(value))

    def _normalize_sizes(self):  # pragma: no cover - not used
        pass

    def _execute_kernel_with_args(self, *a, **k):  # pragma: no cover
        pass

    def _handle_async_completion(self):  # pragma: no cover
        pass

    def _handle_sync_completion(self, extra_kwargs):  # pragma: no cover
        pass


def synthetic_args(kernel: KernelSpec, shape: tuple[int, ...]) -> list[Any]:
    args: list[Any] = []
    for p in kernel.parameters:
        if p.derived:
            continue
        if p.hint:
            args.append(np.zeros(shape, dtype=getattr(np, p.np_dtype)))
        else:
            args.append(SCALAR_VALUE)
    return args


def _describe_arg(a: Any) -> dict[str, Any]:
    if isinstance(a, np.ndarray):
        return {"shape": list(a.shape), "dtype": a.dtype.name}
    return {"scalar": a}


def derive_kernel_args(kernel: KernelSpec, args: list[Any]) -> list[dict[str, Any]]:
    """
    Mirror ``prepare_arguments`` from the backend kernel classes, without a device.

    Returns one record per kernel parameter, in kernel-argument order.
    """
    deriver = _Deriver()
    deriver.params = kernel.parameters
    deriver.map_arguments()
    assert len(deriver.posbyname) == len(args), (kernel.name, len(args))
    out: list[dict[str, Any]] = []
    for param in kernel.parameters:
        dtype = getattr(np, param.np_dtype)
        iarg = deriver.posbyname[param.derived_from]
        arg = args[iarg]
        if not param.hint and not param.dim:
            out.append(
                {"param": param.name, "kind": "scalar", "value": dtype(arg).item()}
            )
        elif param.dim:
            handler = getattr(deriver, f"handle_{param.dim}")
            out.append(
                {
                    "param": param.name,
                    "kind": "dim",
                    "dim": param.dim,
                    "arg": iarg,
                    "value": handler(arg, dtype),
                }
            )
        elif param.hint == "out_like":
            like = np.empty_like(arg, dtype=dtype)
            out.append(
                {
                    "param": param.name,
                    "kind": "out_like",
                    "arg": iarg,
                    "shape": list(like.shape),
                    "dtype": like.dtype.name,
                }
            )
        else:
            out.append(
                {"param": param.name, "kind": "buffer", "role": param.hint, "arg": iarg}
            )
    return out


def default_global_sizes(args: list[Any]) -> dict[str, list[int] | None]:
    """Per-backend default global size when neither global_size nor dispatch_size is given."""
    # ocl/kernel.py and cu/kernel.py: first argument with a .size attribute, flattened
    flat = None
    for a in args:
        if hasattr(a, "size"):
            flat = [int(a.size)]
            break
    # metallic/kernel.py: first ndarray; (shape[0]*shape[1], 1) when 2-D or more
    metal = None
    for a in args:
        if isinstance(a, np.ndarray):
            shape = a.shape
            metal = [shape[0] * shape[1], 1] if len(shape) > 1 else [int(shape[0])]
            break
    return {"ocl": flat, "cuda": flat, "metal": metal}


def cuda_grid_block(
    global_size: list[int], local_size: list[int] | None = None
) -> dict[str, list[int]]:
    """cu/kernel.py _calculate_grid_block."""
    if local_size is None:
        block = [256, 1, 1]
    else:
        block = (list(local_size) + [1, 1])[:3]
    gs = (list(global_size) + [1, 1])[:3]
    grid = [(gs[i] + block[i] - 1) // block[i] for i in range(3)]
    return {"grid": grid, "block": block}


def metal_grid_tgs(
    global_size: list[int], local_size: list[int] | None = None
) -> dict[str, list[int]]:
    """metallic/kernel.py _normalize_sizes plus pymetallic's default threadgroup size."""
    grid = (list(global_size) + [1, 1])[:3]
    if local_size is None:
        tgs = [min(max(1, grid[0]), 256), 1, 1]
    else:
        tgs = (list(local_size) + [1, 1])[:3]
    return {"grid": grid, "tgs": tgs}


def kernel_fixture(kernel: KernelSpec) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    for case, shape in CASES.items():
        args = synthetic_args(kernel, shape)
        record: dict[str, Any] = {"args": [_describe_arg(a) for a in args]}
        try:
            record["kernel_args"] = derive_kernel_args(kernel, args)
        except IndexError as e:
            # e.g. dim_1_of on a 1-D array: Python raises, and so must a port
            record["error"] = f"{type(e).__name__}: {e}"
            cases[case] = record
            continue
        traits = kernel.traits or {}
        if "dispatch_size" in traits:
            try:
                gs = evaluate_dispatch_size(
                    traits["dispatch_size"], tuple(args), kernel.parameters
                )
                record["dispatch_global_size"] = [int(g) for g in gs]
            except (ValueError, IndexError) as e:
                record["dispatch_global_size"] = None
                record["dispatch_error"] = str(e)
        defaults = default_global_sizes(args)
        record["default_global_size"] = defaults
        effective = record.get("dispatch_global_size") or defaults["ocl"]
        if effective:
            record["cuda_launch"] = cuda_grid_block(effective)
        metal_effective = record.get("dispatch_global_size") or defaults["metal"]
        if metal_effective:
            record["metal_launch"] = metal_grid_tgs(metal_effective)
        cases[case] = record
    return {"symbol": kernel.alias or kernel.name, "cases": cases}


def check_dispatch_ast(kernel: KernelSpec, ast: list[dict[str, Any]] | None) -> None:
    """Assert the manifest AST evaluates like dispatch_expr for every synthetic case."""
    traits = kernel.traits or {}
    if ast is None or "dispatch_size" not in traits:
        return
    for shape in CASES.values():
        args = tuple(synthetic_args(kernel, shape))
        try:
            expected = evaluate_dispatch_size(
                traits["dispatch_size"], args, kernel.parameters
            )
        except (ValueError, IndexError):
            continue
        got = tuple(int(evaluate_dispatch_ast(node, args)) for node in ast)
        assert got == tuple(int(e) for e in expected), (
            kernel.name,
            shape,
            got,
            expected,
        )


def derivation_fixtures(manifest: Manifest) -> dict[str, Any]:
    interfaces: dict[str, Any] = {}
    for entry in manifest.interfaces:
        spec: InterfaceSpec = get_interface(entry.interface_text)
        by_name = {k.name: k for k in spec.kernels}
        kernels: dict[str, Any] = {}
        for kentry in entry.kernels:
            kernel = by_name[kentry.name]
            check_dispatch_ast(kernel, kentry.dispatch_size)
            kernels[kentry.name] = kernel_fixture(kernel)
        interfaces[entry.id] = kernels
    return {
        "schema_version": manifest.schema_version,
        "cases": {k: list(v) for k, v in CASES.items()},
        "scalar_value": SCALAR_VALUE,
        "interfaces": interfaces,
    }


def write_derivation_fixtures(out_dir: str, manifest: Manifest) -> str:
    fixtures_dir = os.path.join(out_dir, "fixtures")
    os.makedirs(fixtures_dir, exist_ok=True)
    path = os.path.join(fixtures_dir, "derivation.json")
    with open(path, "w") as f:
        json.dump(derivation_fixtures(manifest), f, indent=1, sort_keys=True)
        f.write("\n")
    return path


__all__ = [
    "CASES",
    "KernelParamSpec",
    "derivation_fixtures",
    "derive_kernel_args",
    "write_derivation_fixtures",
]
