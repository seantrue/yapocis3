#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.
"""
Kernel bundle export: everything a non-Python runtime needs to call yapocis kernels.

A bundle is a directory::

    manifest.json          schema + per-interface kernel signatures and source references
    ocl/<id>.cl            rendered OpenCL C source per interface variant
    metal/<id>.metal       rendered Metal Shading Language source
    cuda/<id>.cu           rendered CUDA C source

where ``<id>`` is the interface name, or ``<name>.<variant>`` for a non-default
variant. The manifest carries the flat runtime model from ``yapocis.core.interface``
(the one the Python backends execute from), so a consumer never parses the IDL or
renders Mako. ``dispatch_size`` trait expressions are pre-parsed into a small AST
(see :func:`parse_dispatch_expr`) with the same semantics as
``yapocis.core.dispatch_expr``.

Schema version history:
    1  initial (2026-10)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import yaml

from yapocis.core.interface import InterfaceSpec, KernelParamSpec, KernelSpec
from yapocis.core.kmgr_base import SOURCE_EXTENSIONS, prepare_program
from yapocis.interfaces import interfaces as list_interfaces

SCHEMA_VERSION = 1
BUNDLE_BACKENDS = ("ocl", "metal", "cuda")

# Backend default build options, mirroring the device managers.
DEFAULT_BUILD_OPTIONS: dict[str, list[str]] = {
    # yapocis/ocl/device_manager.py
    "ocl": [
        "-w",
        "-cl-fast-relaxed-math",
        "-cl-mad-enable",
        "-cl-no-signed-zeros",
        "-cl-finite-math-only",
    ],
    # pymetallic compiles with default MTLCompileOptions
    "metal": [],
    # yapocis/cu/device_manager.py passes nvcc flags; these are the NVRTC spellings
    # (nvcc: --use_fast_math -Xcudafe --diag_suppress=N).
    "cuda": [
        "--use_fast_math",
        "--diag-suppress=177",
        "--diag-suppress=68",
        "--diag-suppress=550",
    ],
}


# ---------------------------------------------------------------------------
# dispatch_size expressions
# ---------------------------------------------------------------------------

_DIM_REF = re.compile(r"(width_of|height_of|depth_of|sizeof|dim_\d+_of)\s+(\w+)")
_TOKEN = re.compile(r"\s*(?:(?P<num>\d+(?:\.\d+)?)|(?P<op>//|[-+*/%()])|(?P<word>\w+))")


class DispatchExprError(ValueError):
    pass


def parse_dispatch_expr(expr: str, params: list[KernelParamSpec]) -> dict[str, Any]:
    """
    Parse one ``dispatch_size`` element into an AST.

    Nodes::

        {"lit": 64}
        {"dim": "dim_1_of", "of": "input", "param_index": 0}
        {"op": "+", "lhs": node, "rhs": node}        ops: + - * / // %
        {"op": "neg", "arg": node}

    ``param_index`` is the index of the first parameter named ``of`` in the full
    parameter list (derived parameters included), which is how
    ``yapocis.core.dispatch_expr`` indexes the *user* argument tuple. The
    evaluator must reproduce that, and the dimension semantics of
    ``dispatch_expr`` (``width_of`` is ``shape[2]`` for 3-D arrays, ``depth_of``
    is ``shape[0]``), not those of the derived-parameter handlers.

    Division ``/`` is true division; the final result is truncated to int.
    """
    text = str(expr).strip()
    tokens = _tokenize(text, params)
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def take():
        nonlocal pos
        t = tokens[pos]
        pos += 1
        return t

    def parse_expr():
        node = parse_term()
        while peek() in ("+", "-"):
            op = take()
            node = {"op": op, "lhs": node, "rhs": parse_term()}
        return node

    def parse_term():
        node = parse_unary()
        while peek() in ("*", "/", "//", "%"):
            op = take()
            node = {"op": op, "lhs": node, "rhs": parse_unary()}
        return node

    def parse_unary():
        if peek() == "-":
            take()
            return {"op": "neg", "arg": parse_unary()}
        if peek() == "+":
            take()
            return parse_unary()
        return parse_atom()

    def parse_atom():
        t = take()
        if t == "(":
            node = parse_expr()
            if take() != ")":
                raise DispatchExprError(f"expected ')' in {expr!r}")
            return node
        if isinstance(t, dict):
            return t
        raise DispatchExprError(f"unexpected token {t!r} in {expr!r}")

    node = parse_expr()
    if pos != len(tokens):
        raise DispatchExprError(f"trailing tokens in {expr!r}")
    return node


def _tokenize(text: str, params: list[KernelParamSpec]) -> list[Any]:
    tokens: list[Any] = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        m = _DIM_REF.match(text, i)
        if m:
            dim, name = m.group(1), m.group(2)
            index = next((k for k, p in enumerate(params) if p.name == name), None)
            if index is None:
                raise DispatchExprError(f"unknown parameter {name!r} in {text!r}")
            tokens.append({"dim": dim, "of": name, "param_index": index})
            i = m.end()
            continue
        m = _TOKEN.match(text, i)
        if not m:
            raise DispatchExprError(f"cannot tokenize {text[i:]!r}")
        if m.group("num") is not None:
            num = m.group("num")
            tokens.append({"lit": float(num) if "." in num else int(num)})
        elif m.group("op") is not None:
            tokens.append(m.group("op"))
        else:
            raise DispatchExprError(f"unexpected word {m.group('word')!r} in {text!r}")
        i = m.end()
    return tokens


def evaluate_dispatch_ast(node: dict[str, Any], args: tuple[Any, ...]) -> float:
    """Evaluate an AST against the user arguments (Python-semantics reference)."""
    import numpy as np

    if "lit" in node:
        return node["lit"]
    if "dim" in node:
        arg = args[node["param_index"]]
        if not isinstance(arg, np.ndarray):
            raise ValueError(f"parameter {node['of']!r} must be an array")
        shape = arg.shape
        dim = node["dim"]
        if dim.startswith("dim_"):
            return shape[int(dim.split("_")[1])]
        if dim == "width_of":
            return shape[2] if len(shape) > 2 else shape[0]
        if dim == "height_of":
            return shape[1] if len(shape) > 1 else 1
        if dim == "depth_of":
            return shape[0]
        if dim == "sizeof":
            return arg.size
        raise ValueError(dim)
    op = node["op"]
    if op == "neg":
        return -evaluate_dispatch_ast(node["arg"], args)
    lhs = evaluate_dispatch_ast(node["lhs"], args)
    rhs = evaluate_dispatch_ast(node["rhs"], args)
    if op == "+":
        return lhs + rhs
    if op == "-":
        return lhs - rhs
    if op == "*":
        return lhs * rhs
    if op == "/":
        return lhs / rhs
    if op == "//":
        return lhs // rhs
    if op == "%":
        return lhs % rhs
    raise ValueError(op)


# ---------------------------------------------------------------------------
# Manifest model
# ---------------------------------------------------------------------------


@dataclass
class ParamEntry:
    name: str
    kind: str  # buffer | scalar | dim | out_like
    dtype: str  # numpy dtype name (float32, int32, complex64, ...)
    ctype: str  # IDL type (float, int, complex64, ...)
    role: str | None = None  # buffer: in | out | inout | resident
    dim: str | None = None  # dim: width_of | height_of | ... | dim_N_of
    of: str | None = None  # dim / out_like: source parameter name
    derived: bool = False


@dataclass
class KernelEntry:
    name: str
    symbol: str
    params: list[ParamEntry]
    user_args: list[str]
    returns: list[str]
    dispatch_size: list[dict[str, Any]] | None
    cuda: dict[str, Any] = field(default_factory=dict)
    traits: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceEntry:
    path: str
    sha256: str
    build_options: list[str]


@dataclass
class InterfaceEntry:
    name: str
    variant: str
    id: str
    interface_text: str
    kernels: list[KernelEntry]
    sources: dict[str, SourceEntry]
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class Manifest:
    schema_version: int
    yapocis_version: str
    backends: list[str]
    interfaces: list[InterfaceEntry]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=False) + "\n"


# ---------------------------------------------------------------------------
# Building entries
# ---------------------------------------------------------------------------

_GLOBAL_SIZE_PARAM = "GLOBAL_SIZE_PARAM"


def takes_global_size(symbol: str, cuda_source: str) -> bool:
    """Same test as yapocis/cu/kernel.py: the kernel signature ends with GLOBAL_SIZE_PARAM."""
    return bool(
        re.search(
            rf"\b{re.escape(symbol)}\s*\([^)]*{_GLOBAL_SIZE_PARAM}\s*\)", cuda_source
        )
    )


def param_entry(p: KernelParamSpec) -> ParamEntry:
    if p.dim:
        return ParamEntry(
            name=p.name,
            kind="dim",
            dtype=p.np_dtype,
            ctype=p.data_type,
            dim=p.dim,
            of=p.derived_from,
            derived=True,
        )
    if p.hint == "out_like":
        return ParamEntry(
            name=p.name,
            kind="out_like",
            dtype=p.np_dtype,
            ctype=p.data_type,
            of=p.derived_from,
            derived=True,
        )
    if p.hint:
        return ParamEntry(
            name=p.name,
            kind="buffer",
            dtype=p.np_dtype,
            ctype=p.data_type,
            role=p.hint,
            derived=False,
        )
    return ParamEntry(
        name=p.name, kind="scalar", dtype=p.np_dtype, ctype=p.data_type, derived=False
    )


def kernel_entry(k: KernelSpec, cuda_source: str | None) -> KernelEntry:
    params = [param_entry(p) for p in k.parameters]
    user_args = [p.name for p in k.parameters if not p.derived]
    # kernel_base.handle_{inout,out,outlike}: these are read back and returned in order
    returns = [p.name for p in k.parameters if p.hint in ("inout", "out", "out_like")]
    symbol = k.alias or k.name
    traits = dict(k.traits or {})
    dispatch = traits.pop("dispatch_size", None)
    dispatch_ast = None
    if dispatch is not None:
        if isinstance(dispatch, (str, int, float)):
            dispatch = [dispatch]
        dispatch_ast = [parse_dispatch_expr(str(e), k.parameters) for e in dispatch]
    cuda: dict[str, Any] = {}
    if cuda_source is not None:
        cuda["takes_global_size"] = takes_global_size(symbol, cuda_source)
    return KernelEntry(
        name=k.name,
        symbol=symbol,
        params=params,
        user_args=user_args,
        returns=returns,
        dispatch_size=dispatch_ast,
        cuda=cuda,
        traits=traits,
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    """Make a template context JSON-serialisable (numpy arrays become lists)."""
    import numpy as np

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


# ---------------------------------------------------------------------------
# Configuration (bundle.yml)
# ---------------------------------------------------------------------------


@dataclass
class BundleConfig:
    variants: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    build_options: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    exclude: list[str] = field(default_factory=list)


def packaged_interface_dir() -> str:
    import yapocis.interfaces

    return os.path.dirname(os.path.abspath(yapocis.interfaces.__file__))


def load_bundle_config(path: str | None = None) -> BundleConfig:
    """Load bundle.yml (next to the packaged interfaces unless a path is given)."""
    if path is None:
        path = os.path.join(packaged_interface_dir(), "bundle.yml")
    if not os.path.exists(path):
        return BundleConfig()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return BundleConfig(
        variants={
            name: {v: (ctx or {}) for v, ctx in (variants or {}).items()}
            for name, variants in (data.get("variants") or {}).items()
        },
        build_options={
            name: {b: list(opts) for b, opts in (per_backend or {}).items()}
            for name, per_backend in (data.get("build_options") or {}).items()
        },
        exclude=list(data.get("exclude") or []),
    )


def packaged_interface_names() -> list[str]:
    """Interfaces shipped in the yapocis package (not YAPOCIS_PROJECT or cwd)."""
    package_dir = packaged_interface_dir()
    names = []
    for entry in sorted(os.listdir(package_dir)):
        if entry.endswith(".interface"):
            names.append(entry[: -len(".interface")])
    return names


def all_interface_names() -> list[str]:
    """Every discoverable interface (package, cwd and YAPOCIS_PROJECT)."""
    return sorted({n.rsplit(".", 1)[0] for n in list_interfaces()})


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def yapocis_version() -> str:
    try:
        from importlib.metadata import version

        return version("yapocis")
    except Exception:  # noqa: BLE001
        return "0.0.0"


def build_interface_entry(
    name: str,
    variant: str,
    context: dict[str, Any],
    backends: tuple[str, ...],
    build_options: dict[str, list[str]] | None = None,
) -> tuple[InterfaceEntry, dict[str, str]]:
    """
    Render one interface variant for the given backends.

    Returns the manifest entry and ``{backend: source text}``.
    """
    sources: dict[str, str] = {}
    interface: InterfaceSpec | None = None
    merged: dict[str, Any] = {}
    for backend in backends:
        interface, merged, src = prepare_program(backend, name, **context)
        sources[backend] = src
    assert interface is not None
    entry_id = name if variant == "default" else f"{name}.{variant}"
    cuda_source = sources.get("cuda")
    kernels = [kernel_entry(k, cuda_source) for k in interface.kernels]
    source_entries: dict[str, SourceEntry] = {}
    for backend, src in sources.items():
        options = (build_options or {}).get(backend, DEFAULT_BUILD_OPTIONS[backend])
        source_entries[backend] = SourceEntry(
            path=f"{backend}/{entry_id}.{SOURCE_EXTENSIONS[backend]}",
            sha256=_sha256(src),
            build_options=list(options),
        )
    from yapocis.core.interface import load_interface
    from yapocis.core.kmgr_base import render_interface_source

    interface_text = render_interface_source(load_interface(name), **merged)
    entry = InterfaceEntry(
        name=interface.name,
        variant=variant,
        id=entry_id,
        interface_text=interface_text,
        kernels=kernels,
        sources=source_entries,
        context=_json_safe(context),
    )
    return entry, sources


def export_bundle(
    out_dir: str,
    names: list[str] | None = None,
    backends: tuple[str, ...] = BUNDLE_BACKENDS,
    config: BundleConfig | None = None,
    include_project: bool = False,
    log=print,
) -> Manifest:
    """
    Write a kernel bundle to ``out_dir``.

    Args:
        out_dir: Target directory (created; existing manifest and sources replaced).
        names: Interface names to export (default: all packaged interfaces).
        backends: Subset of ``BUNDLE_BACKENDS``.
        config: Variants and build options (default: the packaged bundle.yml).
        include_project: Also export interfaces found via YAPOCIS_PROJECT / cwd.
        log: Progress callback.
    """
    config = config or load_bundle_config()
    if names is None:
        names = all_interface_names() if include_project else packaged_interface_names()
    names = [n for n in names if n not in config.exclude]

    os.makedirs(out_dir, exist_ok=True)
    for backend in backends:
        os.makedirs(os.path.join(out_dir, backend), exist_ok=True)

    entries: list[InterfaceEntry] = []
    for name in names:
        variants = config.variants.get(name, {"default": {}})
        for variant, context in variants.items():
            try:
                entry, sources = build_interface_entry(
                    name, variant, context, backends, config.build_options.get(name)
                )
            except Exception as e:  # noqa: BLE001
                log(f"{name}[{variant}]: SKIP {type(e).__name__}: {e}")
                continue
            for backend, src in sources.items():
                with open(os.path.join(out_dir, entry.sources[backend].path), "w") as f:
                    f.write(src)
            entries.append(entry)
            log(f"{entry.id}: {len(entry.kernels)} kernels, {', '.join(sources)}")

    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        yapocis_version=yapocis_version(),
        backends=list(backends),
        interfaces=entries,
    )
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        f.write(manifest.to_json())
    return manifest


def load_manifest(path: str) -> dict[str, Any]:
    """Read a manifest.json (as plain dicts) and check its schema version."""
    with open(path) as f:
        data = json.load(f)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported bundle schema {data.get('schema_version')} (want {SCHEMA_VERSION})"
        )
    return data
