#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

# Process an improper subset of the CORBA IDL grammar to facilitate calling opencl from a python Numpy framework
# Inspired by an IDL parser by Paul McGuire, shipped as a demo with pyparser
#

from dataclasses import dataclass
from typing import Any, Optional

from yapocis.interfaces import find_interface
from .interface_parser import parse_interface_text, ParseException

@dataclass
class KernelParamSpec:
    name: str
    hint: str
    dim: str
    pointer: str
    data_type: str
    np_dtype: str
    derived: bool
    derived_from: str

@dataclass
class KernelSpec:
    name: str
    alias: str | None
    parameters: list[KernelParamSpec]
    definitions: dict[str, KernelParamSpec]
    tiled: bool

@dataclass
class InterfaceSpec:
    name: str
    tiled: bool
    kernels: list[KernelSpec]
    aliases: dict[str, str]

# Public alias used by other modules
Interface = InterfaceSpec

# Map from OpenCL datatype to a numpy data type
ctype_to_nptype = {
    "int": "int32",
    "short": "int16",
    "long": "int64",
    "float": "float32",
    "half": "float16",
    "double": "float64",
}
nptype_to_ctype = {value: key for (key, value) in ctype_to_nptype.items()}

def scan_kernel_parameters(kernel):
    definitions = {
        param.name: param
        for param in kernel.parameters
        if param.hint not in ("out", "out_like")
    }
    return definitions

_interfaces: dict[str, Any] = {}

# ... existing code ...
def get_interfaces(s: str) -> list[InterfaceSpec]:
    """
    Builds an InterfaceSpec for the source interface definition.
    """
    try:
        interfaces = _interfaces[s]
        return interfaces
    except KeyError:
        interfaces = []
    try:
        # Normalize legacy spellings to match the parser's expected tokens.
        # This preserves compatibility with existing tests/specs that use widthof/heightof/outlike/in_out.
        def _normalize(text: str) -> str:
            import re

            subs = [
                (r"\bwidthof\b", "width_of"),
                (r"\bheightof\b", "height_of"),
                (r"\boutlike\b", "out_like"),
                (r"\bin_out\b", "inout"),
                (r"\bdepthof\b", "depth_of"),
            ]
            for pat, repl in subs:
                text = re.sub(pat, repl, text)
            return text

        s_norm = _normalize(s)
        parsed_interfaces = parse_interface_text(s_norm)
    except ParseException as e:
        print(e.args[0], e.args[2], e.line)
        source = e.args[0].splitlines()
        cursor = " " * (e.col - 1)
        cursor += "^"
        source.insert(e.lineno, cursor)
        source = "\n".join(source)
        raise RuntimeError(f"Can't parse:\n{source}")
    out: list[InterfaceSpec] = []
    for p_iface in parsed_interfaces:
        kernel_definitions: list[KernelSpec] = []
        kernel_aliases: dict[str, str] = {}
        interface_tiled = bool(getattr(p_iface, "meta", {}).get("tiled", False))

        for p_kernel in p_iface.kernels:
            aliasof = getattr(p_kernel, "alias_of", None)
            extends = getattr(p_kernel, "extends", None)

            # Build parameter definitions first (non-output parameters define dtypes)
            temp_params: dict[str, KernelParamSpec] = {}
            for p in p_kernel.params:
                role = (p.role or "").lower() if hasattr(p, "role") else ""
                name = p.name
                dtype = (p.dtype or "") if hasattr(p, "dtype") else ""
                hint = ""
                dim = ""
                pointer = ""
                if role in ("in", "out", "inout", "in_out", "resident", "readwrite"):
                    hint = "inout" if role in ("inout", "in_out", "readwrite") else role
                    pointer = "*"
                elif role in ("out_like", "outlike"):
                    hint = "out_like"
                    pointer = "*"
                    dtype = ""  # derive later from like param
                elif role in (
                    "size_of",
                    "sizeof",
                    "width_of",
                    "widthof",
                    "height_of",
                    "heightof",
                    "depth_of",
                ):
                    dim = {
                        "size_of": "size_of",
                        "sizeof": "sizeof",
                        "width_of": "width_of",
                        "widthof": "width_of",
                        "height_of": "height_of",
                        "heightof": "height_of",
                        "depth_of": "depth_of",
                    }[role]
                    dtype = "int"
                # scalar form: dtype name
                elif dtype:
                    hint = ""
                np_dtype = ctype_to_nptype.get(dtype, dtype)
                # Only record definitions for non-output, non-derived params
                if not dim and hint not in ("out", "out_like"):
                    temp_params[name] = KernelParamSpec(
                        name=name,
                        hint=hint,
                        dim=dim,
                        pointer=pointer,
                        data_type=dtype,
                        np_dtype=np_dtype,
                        derived=False,
                        derived_from=name,
                    )

            # Now finalize kernel parameters list including derived forms
            kernel_parameters: list[KernelParamSpec] = []
            for p in p_kernel.params:
                role = (p.role or "").lower() if hasattr(p, "role") else ""
                name = p.name
                dtype = (p.dtype or "") if hasattr(p, "dtype") else ""
                if role in (
                    "size_of",
                    "sizeof",
                    "width_of",
                    "widthof",
                    "height_of",
                    "heightof",
                    "depth_of",
                ):
                    dim = {
                        "size_of": "size_of",
                        "sizeof": "sizeof",
                        "width_of": "width_of",
                        "widthof": "width_of",
                        "height_of": "height_of",
                        "heightof": "height_of",
                        "depth_of": "depth_of",
                    }[role]
                    kernel_parameters.append(
                        KernelParamSpec(
                            name=name,
                            hint="",
                            dim=dim,
                            pointer="",
                            data_type="int",
                            np_dtype=ctype_to_nptype.get("int", "int32"),
                            derived=True,
                            derived_from=name,
                        )
                    )
                elif role in ("out_like", "outlike"):
                    like_dtype = temp_params[name].data_type if not dtype else dtype
                    kernel_parameters.append(
                        KernelParamSpec(
                            name=f"out_like_{name}",
                            hint="out_like",
                            dim="",
                            pointer="*",
                            data_type=like_dtype,
                            np_dtype=ctype_to_nptype.get(like_dtype, like_dtype),
                            derived=True,
                            derived_from=name,
                        )
                    )
                else:
                    # in/out/inout/resident, or scalar (no role)
                    hint = ""
                    pointer = ""
                    if role in (
                        "in",
                        "out",
                        "inout",
                        "in_out",
                        "resident",
                        "readwrite",
                    ):
                        hint = (
                            "inout"
                            if role in ("inout", "in_out", "readwrite")
                            else role
                        )
                        pointer = "*"
                    np_dtype = ctype_to_nptype.get(dtype, dtype)
                    kernel_parameters.append(
                        KernelParamSpec(
                            name=name,
                            hint=hint,
                            dim="",
                            pointer=pointer,
                            data_type=dtype,
                            np_dtype=np_dtype,
                            derived=False,
                            derived_from=name,
                        )
                    )

            param_definitions = {
                kp.name: kp
                for kp in kernel_parameters
                if kp.hint not in ("out", "out_like")
            }

            base_name = (
                getattr(p_kernel, "name", None)
                or getattr(p_kernel, "kname", None)
                or ""
            )
            # For extensions, expose as "<base>_<new>" but call underlying base kernel
            if extends:
                kernel_name = f"{extends}_{base_name}"
                alias_for_runtime = extends
            else:
                kernel_name = base_name
                alias_for_runtime = aliasof
            kernel_tiled = bool(
                getattr(p_kernel, "traits", {}).get("tiled", False) or interface_tiled
            )
            kernel_spec = KernelSpec(
                name=kernel_name,
                alias=alias_for_runtime,
                parameters=kernel_parameters,
                definitions=param_definitions,
                tiled=kernel_tiled,
            )
            kernel_definitions.append(kernel_spec)
            if aliasof:
                kernel_aliases[kernel_name] = aliasof

        out.append(
            InterfaceSpec(
                name=p_iface.name,
                tiled=interface_tiled,
                kernels=kernel_definitions,
                aliases=kernel_aliases,
            )
        )
    _interfaces[s] = out
    return out

# ... existing code ...

def get_interface(s: str) -> InterfaceSpec:
    return get_interfaces(s)[0]

def load_interface(interface_spec: str) -> str:
    """
    Find an interface, or return source if it is an interface spec.
    Args:
        interface_spec: either an interface name or a spec

    Returns:
        an interface specification
    """
    if interface_spec.strip().startswith("interface "):
        return interface_spec

    path = find_interface(interface_spec)
    if path is None:
        raise RuntimeError(f"Can't find {interface_spec}")
    with open(path) as inf:
        return inf.read()
