#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interface_parser.py  —  PyParsing-only interface → IR

Parses extended `.interface` specs into a backend-agnostic IR suitable for stub
generation. This file has **no** regex fallback; PyParsing is a hard requirement.

Supported language features
---------------------------
File-level:
  interface <Name> {
    meta {
      layout <planar|NHWC|NCHW>;
      dtype <float32|...>;
      fast_math <on|off|true|false>;
      workgroup (<x>,<y>,<z>);
      class <elementwise|stencil2d|separable|reduction|graphics_2d>;
      blend <alpha|src_over|additive>;
      cull <none|tile>;
      binning <on|off>;
      tile_pixels (<w>,<h>);
      stage_tile <on|off>;
    }

    # kernels, aliases, and extensions in any order …

    kernel <name>( <params> );
    [with { <traits>; [backend <b> { <overrides>; }]* }];

    alias <base> as <new> ;
    alias <base> as <new>( <params> ) ;

    extension <new> of <base>( <params> )
    [with { <traits>; [backend <b> { <overrides>; }]* }];

  };

Parameters:
  - qualifiers (any order, optional):  constant | texture2d | local bytes_of(<expr>)
  - role: in | out | resident | out_like | READWRITE
  - dtype: float|float32|half|f16|double|int|int32|uint32|short|int16|uint16|boolean|char|float4
  - name: identifier
  - optional fixed shape: [ ... ]  (kept as string, e.g. "[9]")
  - optional layout suffix: : layout <tag>

Traits (within `with { ... }` blocks):
  class, axis, radius, boundary, tile, halo, fastmem, taps, identity, op,
  deterministic, accumulator_dtype, texture_input, vector_width, blend, cull

Backend overrides:
  backend <name> { <same-style key value; pairs> }

IR model:
  InterfaceSpec { name, meta: dict, kernels: [Kernel] }
  Kernel { name, params: [Param], shapes: dict(str->str), traits: dict,
           backend_overrides: dict(str->dict), alias_of: str | None, extends: str | None }
  Param  { role, dtype, name, shape: str | None, layout: str | None, qualifiers: dict }

Author: OpenCL↔Metal stub generator project
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pyparsing import (
    Word, alphas, alphanums, nums,
    oneOf, Keyword, Suppress, Optional as PPOptional, Group, delimitedList,
    ZeroOrMore, Regex, ParseException, ParserElement, Literal,
    cppStyleComment, cStyleComment, pythonStyleComment
)

# ---------------------------------------------------------------------------
# IR dataclasses
# ---------------------------------------------------------------------------

@dataclass
class KernelParamSpec:
    name: str
    role: str | None = None
    dtype: str | None = None
    shape: str | None = None           # e.g., "[9]"
    layout: str | None = None          # e.g., "planar"
    qualifiers: dict[str, str] = field(default_factory=dict)  # {"constant":"true","local":"bytes_of(...)","texture2d":"true"}

@dataclass
class KernelSpec:
    name: str
    params: list[KernelParamSpec]
    shapes: dict[str, str]                        # (reserved for width_of/height_of normalization if needed)
    traits: dict[str, Any]
    backend_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Provenance (not used by generator, but useful for tooling)
    alias_of: str | None = None
    extends: str | None = None

@dataclass
class InterfaceSpec:
    name: str
    meta: dict[str, Any]
    kernels: list[KernelSpec]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _items_to_dict(items: list[tuple[str, Any]]) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for k, v in items:
        if isinstance(v, list):
            v = tuple(v)
        d[k] = v
    return d

def _norm_bool(v: Any, default: bool=False) -> bool:
    if isinstance(v, bool): return v
    if v is None: return default
    s = str(v).lower()
    return s in ("on", "true", "yes", "1")

def _pp_to_param_list(pp_params) -> list[KernelParamSpec]:
    out: list[KernelParamSpec] = []
    for p in (pp_params or []):
        quals: dict[str, str] = {}
        for q in (p.quals or []):
            if "bytes" in q:
                quals["local"] = f"bytes_of({q.bytes})"
            else:
                quals[q.q] = "true"
        shape = f"[{p.shape.shape}]" if p.shape and "shape" in p.shape else None
        layout = p.layout if ("layout" in p and p.layout) else None
        core = p[1] if len(p) > 1 else p
        role = core.get("role", None)
        dtype_val = core.get("dtype", None)
        name_val = core.get("name", None)
        out.append(KernelParamSpec(name=name_val, role=role, dtype=dtype_val,
                                   shape=shape, layout=layout, qualifiers=quals))
    return out

# ---------------------------------------------------------------------------
# Grammar
# ---------------------------------------------------------------------------

ParserElement.set_default_whitespace_chars(" \t\r\n")

ident    = Word(alphas + "_", alphanums + "_")
integer  = Regex(r"-?\d+").setParseAction(lambda t: int(t[0]))
number   = Regex(r"-?\d+(?:\.\d+)?").setParseAction(lambda t: float(t[0]))

lparen, rparen = Suppress("("), Suppress(")")
lbrace, rbrace = Suppress("{"), Suppress("}")
colon, comma, semi = Suppress(":"), Suppress(","), Suppress(";")

tuple2 = Group(lparen + number + comma + number + rparen)
tuple3 = Group(lparen + number + comma + number + comma + number + rparen)

# --- meta { ... } ----------------------------------------------------------

meta_key = oneOf("layout dtype fast_math workgroup class blend cull binning tile_pixels stage_tile")
meta_val = (ident | number | tuple2 | tuple3)
meta_kv  = Group(meta_key("k") + meta_val("v") + semi)
meta_block = PPOptional(Keyword("meta") + lbrace + Group(ZeroOrMore(meta_kv))("items") + rbrace)

# --- qualifiers & parameter forms -----------------------------------------

qual_local  = Group(Keyword("local")("q") + Keyword("bytes_of") + lparen + Regex(r"[^)]*")("bytes") + rparen)
qual_simple = oneOf("constant texture2d")("q")
qualifier   = (qual_local | qual_simple)

layout_suffix = PPOptional(colon + Keyword("layout") + ident("layout"))

dtype = oneOf("float float32 half f16 double int int32 uint32 short int16 uint16 boolean char float4 complex64 complex32")

# Roles and legacy spellings
io_role_kw   = oneOf("in out inout in_out resident READWRITE")
dim_role_kw  = oneOf("size_of sizeof width_of widthof height_of heightof depth_of depthof")
outlike_role = oneOf("out_like outlike")

brackets = Group(Suppress("[") + Regex(r"[^\]]*")("shape") + Suppress("]"))

# Parameter patterns
param_core = (
    Group((outlike_role | dim_role_kw)("role") + PPOptional(dtype("dtype")) + ident("name"))
    |
    Group(io_role_kw("role") + dtype("dtype") + ident("name"))
    |
    Group(dtype("dtype") + ident("name"))
)

param = Group(
    Group(ZeroOrMore(qualifier))("quals") +
    param_core +
    PPOptional(brackets)("shape") +
    layout_suffix
)

# --- kernel & with { ... } -------------------------------------------------

kernel_kw = Keyword("kernel")
kernel_sig = (kernel_kw + PPOptional(Keyword("tiled")("tiled")) + ident("kname") +
              lparen + PPOptional(delimitedList(param))("params") + rparen + semi)

trait_key = oneOf("""class axis radius boundary tile halo fastmem taps identity op
                     deterministic accumulator_dtype texture_input vector_width blend cull""")
val_token = (ident | number | tuple2 | tuple3 | Group(lparen + integer + comma + integer + rparen))
kv_pair   = Group(trait_key("k") + val_token("v") + semi)

backend_block = Group(Keyword("backend") + ident("backend") + lbrace +
                      Group(ZeroOrMore(kv_pair))("items") + rbrace)

with_block = PPOptional(Keyword("with") + lbrace +
                        Group(ZeroOrMore(kv_pair))("traits") +
                        Group(ZeroOrMore(backend_block))("backends") +
                        rbrace + semi)

kernel_stmt = Group(kernel_sig("sig") + with_block("withs"))

# --- alias (with optional signature) --------------------------------------

alias_stmt = Group(
    Keyword("alias") + ident("base") + Keyword("as") + ident("new") +
    PPOptional(lparen + PPOptional(delimitedList(param))("params") + rparen) + semi
)

# --- extension (with optional with { ... }) -------------------------------

ext_stmt = Group(
    Keyword("extension")("extension") + ident("new") + (Keyword("of") | Keyword("as")) + ident("base") +
    lparen + PPOptional(delimitedList(param))("params") + rparen +
    PPOptional(Keyword("with") + lbrace +
               Group(ZeroOrMore(kv_pair))("traits") +
               Group(ZeroOrMore(backend_block))("backends") +
               rbrace) + semi)

# --- interface -------------------------------------------------------------

iface_kw = Keyword("interface")
interface_grammar = (
    iface_kw + PPOptional(Keyword("tiled")("tiled")) + ident("iname") + lbrace +
    meta_block("meta") +
    Group(ZeroOrMore(kernel_stmt | alias_stmt | ext_stmt))("items") +
    rbrace + PPOptional(semi)
)

interfaces_grammar = Group(interface_grammar)
# Ignore C/C++ style comments throughout
interfaces_grammar.ignore(cppStyleComment)
interfaces_grammar.ignore(cStyleComment)
interfaces_grammar.ignore(pythonStyleComment)
# ---------------------------------------------------------------------------
# Parse → IR
# ---------------------------------------------------------------------------

def parse_interface_text(src: str) -> list[InterfaceSpec]:
    """
    Parse `.interface` text and return an InterfaceSpec with all kernels,
    including aliases and extensions **materialized** into concrete kernels.
    """
    interface_specs: list[InterfaceSpec] = []
    try:
        reses = interfaces_grammar.parseString(src, parseAll=True)
    except ParseException as e:
        # Re-raise with a slightly clearer message
        raise ParseException(f"[interface_parser] Parse error at line {e.lineno}, col {e.col}: {e.msg}\n{e.line}") from e
    for res in reses:
        # ---- meta
        meta_items: list[tuple[str, Any]] = []
        if "meta" in res and res.meta and "items" in res.meta:
            for kv in res.meta["items"]:
                v = kv.v.asList() if hasattr(kv.v, "asList") else kv.v
                meta_items.append((kv.k, v))
        md = _items_to_dict(meta_items)
        meta: dict[str, Any] = {
            "layout":   md.get("layout", "planar"),
            "dtype":    md.get("dtype", "float32"),
            "fast_math": _norm_bool(md.get("fast_math", "on"), True),
            "workgroup": tuple(md.get("workgroup", (32, 8, 1))),
        }
        # Propagate 'tiled' modifier at the interface level via meta for downstream consumers
        if "tiled" in res and res.tiled:
            meta["tiled"] = True
        # optional meta extras
        for k in ("class","blend","cull","binning","tile_pixels","stage_tile"):
            if k in md:
                meta[k] = md[k]

        # ---- first pass: collect base kernels; defer alias/extension
        kernels: list[KernelSpec] = []
        kernel_by_name: dict[str, KernelSpec] = {}
        pending_aliases: list[tuple[str, str, list[KernelParamSpec] | None]] = []
        pending_exts: list[tuple[str, str, list[KernelParamSpec], dict[str,Any], dict[str,dict[str,Any]]]] = []

        for item in res["items"]:
            # Kernel
            if "sig" in item:
                sig = item.sig
                params = _pp_to_param_list(item.params)
                traits: dict[str, Any] = {}
                backends: dict[str, dict[str, Any]] = {}

                if "withs" in item and item.withs:
                    wt = item.withs
                    if "traits" in wt and wt.traits:
                        for kv in wt.traits:
                            v = kv.v.asList() if hasattr(kv.v, "asList") else kv.v
                            traits[kv.k] = v
                    if "backends" in wt and wt.backends:
                        for bb in wt.backends:
                            items = {}
                            for kv in bb["items"]:
                                v = kv.v.asList() if hasattr(kv.v, "asList") else kv.v
                                items[kv.k] = v
                            backends[bb.backend] = items
                # kernel-level tiled modifier
                if ("tiled" in item and item.tiled) or ("tiled" in sig and sig.tiled):
                    traits["tiled"] = True

                k = KernelSpec(
                    name=item.kname,
                    params=params,
                    shapes={},         # width_of/height_of symbols can be normalized later if desired
                    traits=traits,
                    backend_overrides=backends
                )
                kernels.append(k)
                kernel_by_name[k.name] = k
                continue

            # Alias
            if "base" in item and "new" in item and not ("extension" in item):
                # Grammar is: alias <new_name> as <base_name>(...)
                new_name, base_name = item.base, item.new
                new_params = _pp_to_param_list(item.params) if "params" in item and item.params else None
                pending_aliases.append((new_name, base_name, new_params))
                continue

            # Extension
            if "new" in item and "base" in item and "params" in item:
                new, base = item.new, item.base
                ex_params = _pp_to_param_list(item.params)
                traits: dict[str, Any] = {}
                backends: dict[str, dict[str, Any]] = {}
                if "traits" in item and item.traits:
                    for kv in item.traits:
                        v = kv.v.asList() if hasattr(kv.v, "asList") else kv.v
                        traits[kv.k] = v
                if "backends" in item and item.backends:
                    for bb in item.backends:
                        items = {}
                        for kv in bb["items"]:
                            v = kv.v.asList() if hasattr(kv.v, "asList") else kv.v
                            items[kv.k] = v
                        backends[bb.backend] = items
                pending_exts.append((base, new, ex_params, traits, backends))
                continue

        # ---- resolve aliases
        for new_name, base_name, new_params in pending_aliases:
            if base_name not in kernel_by_name:
                raise ParseException(f"[interface_parser] alias refers to unknown kernel '{base_name}'")
            src = kernel_by_name[base_name]
            kcopy = KernelSpec(
                name=new_name,
                params=(new_params if new_params else [KernelParamSpec(**vars(p)) for p in src.params]),
                shapes=dict(src.shapes),
                traits=dict(src.traits),
                backend_overrides={b: dict(v) for b, v in src.backend_overrides.items()},
                alias_of=base_name
            )
            kernels.append(kcopy)
            kernel_by_name[new_name] = kcopy

        # ---- resolve extensions
        def _merge_traits(base_traits: dict[str,Any], add_traits: dict[str,Any]) -> dict[str,Any]:
            out = dict(base_traits)
            out.update(add_traits or {})
            return out

        for base, new, ex_params, add_traits, add_backends in pending_exts:
            if base not in kernel_by_name:
                raise ParseException(f"[interface_parser] extension refers to unknown kernel '{base}'")
            src = kernel_by_name[base]
            be_over = {k: dict(v) for k, v in src.backend_overrides.items()}
            be_over.update(add_backends or {})
            kext = KernelSpec(
                name=new,
                params=(ex_params if ex_params else [KernelParamSpec(**vars(p)) for p in src.params]),
                shapes={},  # can be derived if needed
                traits=_merge_traits(src.traits, add_traits),
                backend_overrides=be_over,
                extends=base
            )
            kernels.append(kext)
            kernel_by_name[new] = kext
        interface_spec = InterfaceSpec(name=res.iname, meta=meta, kernels=kernels)
        interface_specs.append(interface_spec)
    return interface_specs
