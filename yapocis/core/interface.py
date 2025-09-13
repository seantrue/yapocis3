#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

# Process an improper subset of the CORBA IDL grammar to facilitate calling opencl from a python Numpy framework
# Inspired by an IDL parser by Paul McGuire, shipped as a demo with pyparser
#

from yapocis.interfaces import find_interface
from yapocis.utils.stereo_typing import Any, Dict, List
from .interface_parser import parse_interface_text, KernelParamSpec, KernelSpec, InterfaceSpec, ParseException

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
        interfaces = parse_interface_text(s)
    except ParseException as e:
        print(e.args[0], e.args[2], e.line)
        source = e.args[0].splitlines()
        cursor = " " * (e.col - 1)
        cursor += "^"
        source.insert(e.lineno, cursor)
        source = "\n".join(source)
        raise RuntimeError(f"Can't parse:\n{source}")
    for interface in interfaces:
        kernel_definitions = []
        kernel_aliases = {}
        interface_tiled = bool(getattr(interface, "tiled", ""))
        for kernel in interface.kernels:
            aliasof = kernel.alias_of
            param_definitions = scan_kernel_parameters(kernel)
            kernel_parameters = []
            for param in kernel.parameters:
                param_name = derived_from = param.name
                derived = False
                if param.dim:
                    derived = True
                    data_type = "int"
                elif param.hint in ("outlike", "out_like"):
                    derived = True
                    param_name = f"out_like_{param_name}"
                    data_type = param_definitions[param.name].data_type if param.data_type == '' else param.data_type
                else:
                    data_type = param.data_type
                np_dtype = ctype_to_nptype.get(data_type, data_type)
                kernel_parameters.append(
                    KernelParamSpec(name=param_name,
                                    hint=param.hint,
                                    dim=param.dim,
                                    pointer=param.pointer,
                                    data_type=data_type,
                                    np_dtype=np_dtype,
                                    derived=derived,
                                    derived_from=derived_from))
            kernel_name = f"{aliasof}_{kernel.name}" if kernel.kernel_type == "extension" else kernel.name
            kernel_tiled = bool(getattr(kernel, "tiled", "")) or interface_tiled
            kernel_spec = KernelSpec(name=kernel_name,
                                     alias=aliasof,
                                     parameters=kernel_parameters,
                                     definitions=param_definitions,
                                     tiled=kernel_tiled)
            kernel_definitions.append(kernel_spec)
            if kernel.alias_of:
                kernel_aliases[kernel_name] = kernel.alias_of
        interfaces.append(InterfaceSpec(name=interface.name,
                                    tiled=interface_tiled,
                                    kernels=kernel_definitions,
                                    aliases=kernel_aliases))
    _interfaces[s] = interfaces
    return interfaces


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
