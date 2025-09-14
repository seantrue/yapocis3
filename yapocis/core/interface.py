#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

# Process an improper subset of the CORBA IDL grammar to facilitate calling opencl from a python Numpy framework
# Inspired by an IDL parser by Paul McGuire, shipped as a demo with pyparser
#

from dataclasses import dataclass
from typing import Optional

from yapocis.interfaces import find_interface
from yapocis.utils.stereo_typing import Any, Dict, List
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
    alias: Optional[str]
    parameters: List[KernelParamSpec]
    definitions: Dict[str, KernelParamSpec]
    tiled: bool


@dataclass
class InterfaceSpec:
    name: str
    tiled: bool
    kernels: List[KernelSpec]
    aliases: Dict[str, str]

# Public alias used by other modules
Interface = InterfaceSpec

# Map from OpenCL datatype to a numpy data type
ctype_to_nptype = {
    "int":    "int32",
    "short":  "int16",
    "long":   "int64",
    "float":  "float32",
    "half":   "float16",
    "double": "float64",
}
nptype_to_ctype = {value: key for (key, value) in ctype_to_nptype.items()}


def scan_kernel_parameters(kernel):
    definitions = {param.name: param for param in kernel.parameters if param.hint not in ("out", "out_like")}
    return definitions

_interfaces: Dict[str, Any] = {}
# ... existing code ...
def get_interfaces(s: str) -> List[InterfaceSpec]:
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
    out: List[InterfaceSpec] = []
    for p_iface in parsed_interfaces:
        kernel_definitions: List[KernelSpec] = []
        kernel_aliases: Dict[str, str] = {}
        interface_tiled = bool(getattr(p_iface, "meta", {}).get("tiled", False))

        for p_kernel in p_iface.kernels:
            aliasof = getattr(p_kernel, "alias_of", None)
            extends = getattr(p_kernel, "extends", None)

            # Build parameter definitions first (non-output parameters define dtypes)
            temp_params: Dict[str, KernelParamSpec] = {}
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
                elif role in ("size_of", "sizeof", "width_of", "widthof", "height_of", "heightof", "depth_of"):
                    dim = {
                        "size_of": "size_of", "sizeof": "sizeof",
                        "width_of": "width_of", "widthof": "width_of",
                        "height_of": "height_of", "heightof": "height_of",
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
                        name=name, hint=hint, dim=dim, pointer=pointer,
                        data_type=dtype, np_dtype=np_dtype, derived=False, derived_from=name,
                    )

            # Now finalize kernel parameters list including derived forms
            kernel_parameters: List[KernelParamSpec] = []
            for p in p_kernel.params:
                role = (p.role or "").lower() if hasattr(p, "role") else ""
                name = p.name
                dtype = (p.dtype or "") if hasattr(p, "dtype") else ""
                if role in ("size_of", "sizeof", "width_of", "widthof", "height_of", "heightof", "depth_of"):
                    dim = {
                        "size_of": "size_of", "sizeof": "sizeof",
                        "width_of": "width_of", "widthof": "width_of",
                        "height_of": "height_of", "heightof": "height_of",
                        "depth_of": "depth_of",
                    }[role]
                    kernel_parameters.append(KernelParamSpec(
                        name=name, hint="", dim=dim, pointer="", data_type="int",
                        np_dtype=ctype_to_nptype.get("int", "int32"), derived=True, derived_from=name,
                    ))
                elif role in ("out_like", "outlike"):
                    like_dtype = temp_params[name].data_type if not dtype else dtype
                    kernel_parameters.append(KernelParamSpec(
                        name=f"out_like_{name}", hint="out_like", dim="", pointer="*",
                        data_type=like_dtype, np_dtype=ctype_to_nptype.get(like_dtype, like_dtype),
                        derived=True, derived_from=name,
                    ))
                else:
                    # in/out/inout/resident, or scalar (no role)
                    hint = ""
                    pointer = ""
                    if role in ("in", "out", "inout", "in_out", "resident", "readwrite"):
                        hint = "inout" if role in ("inout", "in_out", "readwrite") else role
                        pointer = "*"
                    np_dtype = ctype_to_nptype.get(dtype, dtype)
                    kernel_parameters.append(KernelParamSpec(
                        name=name, hint=hint, dim="", pointer=pointer,
                        data_type=dtype, np_dtype=np_dtype, derived=False, derived_from=name,
                    ))

            param_definitions = {kp.name: kp for kp in kernel_parameters if kp.hint not in ("out", "out_like")}

            base_name = getattr(p_kernel, "name", None) or getattr(p_kernel, "kname", None) or ""
            # For extensions, expose as "<base>_<new>" but call underlying base kernel
            if extends:
                kernel_name = f"{extends}_{base_name}"
                alias_for_runtime = extends
            else:
                kernel_name = base_name
                alias_for_runtime = aliasof
            kernel_tiled = bool(getattr(p_kernel, "traits", {}).get("tiled", False) or interface_tiled)
            kernel_spec = KernelSpec(name=kernel_name,
                                     alias=alias_for_runtime,
                                     parameters=kernel_parameters,
                                     definitions=param_definitions,
                                     tiled=kernel_tiled)
            kernel_definitions.append(kernel_spec)
            if aliasof:
                kernel_aliases[kernel_name] = aliasof

        out.append(InterfaceSpec(name=p_iface.name, tiled=interface_tiled,
                                  kernels=kernel_definitions, aliases=kernel_aliases))
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


def explain_interface(interface, file=None):
    print("Interface:", interface.name, file=file)
    for kernel in interface.kernels:
        tiling = "tiled" if kernel.tiled else "untiled"
        if kernel.alias:
            print(f"Kernel: {kernel.name} {tiling} alias for '{kernel.alias}'", file=file)
        else:
            print(f"Kernel: {kernel.name} {tiling}", file=file)
        iparam = 0
        ireturn = 0

        def motion(param):
            to_gpu = param.hint in ("in", "inout", "in_out")
            from_gpu = param.hint in ("out", "inout", "out_like", "outlike")
            on_gpu = param.hint in ("resident", "working")
            if to_gpu:
                print("  Before call, copy buffer from array to GPU", file=file)
            if from_gpu:
                print("  After call, copy buffer from GPU into array", file=file)
            if on_gpu:
                print("  Buffer already on GPU", file=file)
            if not from_gpu:
                pointer = "*" if param.hint else ""
                print(f"  Pass as parameter of type {pointer}{param.np_dtype}", file=file)

        def returned(param, ireturn):
            print(f"  Return as {param.np_dtype} (tuple position {ireturn})", file=file)
            return ireturn + 1
        def format_param(param):
            return f"{param.data_type} {param.pointer+' ' if param.pointer else ''}{param.name}"
        for param in kernel.parameters:
            visible_param = not param.derived
            if visible_param:
                print(f"  Param {iparam}: {format_param(param)}", file=file)
                iparam += 1
            else:
                print(f"  Param derived: {format_param(param)}", file=file)
            if param.hint:
                if param.hint == "resident":
                    print(f"  Buffer is resident on GPU", file=file)
                    motion(param)
                elif param.hint in ("out_like", "outlike"):
                    param_name = param.name.removeprefix("out_like_")
                    param_name = param_name.removeprefix("outlike_")
                    print(
                        f"  Allocate an array like '{param_name}' (opencl arg position={iparam}, dtype={param.np_dtype})",
                        file=file)
                    motion(param)
                    ireturn = returned(param, ireturn)
                elif param.hint == "out":
                    print(f"  Allocate an array for {param.name}", file=file)
                    motion(param)
                    ireturn = returned(param, ireturn)
                else:
                    print(f"  Coerce '{param.name}' to {param.np_dtype}", file=file)
                    motion(param)
            if param.dim:
                print(f"  Pass in as {param.dim} `{param.name}`", file=file)
                motion(param)
            print(file=file)
        print(file=file)


def opencl_stubs(interface, file=None):
    print("// kernels", interface.name, file=file)
    for kernel in interface.kernels:
        if kernel.alias:
            continue
        parameter_definitions = scan_kernel_parameters(kernel)
        kernel_parameters = []
        for parameter in kernel.parameters:
            if parameter.hint == "out_like":
                kernel_parameters.append(f"__global {parameter.data_type} *out_like_{parameter.name}")
            elif parameter.hint:
                kernel_parameters.append(f"__global {parameter.data_type} *{parameter.name}")
            elif parameter.dim:
                kernel_parameters.append(f"int {parameter.dim}_{parameter.name}")
            else:
                kernel_parameters.append(f"{parameter.data_type} {parameter.name}")
        print(f"__kernel void {kernel.name}({', '.join(kernel_parameters)}) {{", file=file)
        print("}", file=file)
    print(file=file)


def python_stubs(interface, file=None):
    print("# interface", interface.name, file=file)
    kernels = interface.kernels
    for kernel in kernels:
        python_parameters = []
        python_returns = []
        for parameter in kernel.parameters:
            if parameter.hint in ("out_like", "out", "outlike"):
                data_type = f"Array[{parameter.np_dtype}]"
                python_returns.append(data_type)
            elif parameter.hint:
                data_type = f"Array[{parameter.np_dtype}]"
                python_parameters.append(f"{parameter.name}: {data_type}")
            elif not parameter.pointer and not parameter.dim:
                python_parameters.append(f"{parameter.name}: {parameter.np_dtype}")
        if python_returns:
            python_returns = ", ".join(python_returns)
        else:
            python_returns = "None"
        print(f"def {kernel.name}({', '.join(python_parameters)}) -> {python_returns}:", file=file)
        print("    pass", file=file)
    print(file=file)
