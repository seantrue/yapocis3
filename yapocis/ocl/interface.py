#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

# Process an improper subset of the CORBA IDL grammar to facilitate calling opencl from a python Numpy framework
# Inspired by an IDL parser by Paul McGuire, shipped as a demo with pyparser
#
from collections import namedtuple

from pyparsing import Optional, ParseException, OneOrMore, Literal, Keyword, Word, alphas, alphanums, \
    delimitedList, Group, ZeroOrMore, restOfLine, cStyleComment
from yapocis.interfaces import find_interface
from yapocis.utils.typing import List, Dict, Any


# Map from OpenCL datatype to a numpy data type


def INTERFACECL_BNF() -> OneOrMore:
    """
    pyparser grammar for the yapocis interface specification.
    Inspired by CORBA IDL parser by Paul McGuire, shipped as a demo with pyparser.
    """
    # punctuation
    lbrace = Literal("{")
    rbrace = Literal("}")
    lparen = Literal("(")
    rparen = Literal(")")
    dot = Literal(".")
    star = Literal("*")
    semi = Literal(";")

    # keywords
    alias_ = Keyword("alias")
    as_ = Keyword("as")
    boolean_ = Keyword("boolean")
    char_ = Keyword("char")
    complex64_ = Keyword("complex64")
    depthof_ = Keyword("depthof") | Keyword("depth_of")
    extension_ = Keyword("extension")
    float_ = Keyword("float")
    float32_ = Keyword("float32")
    heightof_ = Keyword("heightof") | Keyword("height_of")
    inout_ = Keyword("inout")
    interface_ = Keyword("interface")
    in_ = Keyword("in")
    int_ = Keyword("int")
    int16_ = Keyword("int16")
    int32_ = Keyword("int32")
    kernel_ = Keyword("kernel")
    of_ = Keyword("of")
    out_ = Keyword("out")
    out_like_ = Keyword("out_like") | Keyword("outlike")
    resident_ = Keyword("resident")
    short_ = Keyword("short")
    sizeof_ = Keyword("sizeof") | Keyword("size_of")
    uint16_ = Keyword("uint16")
    uint32_ = Keyword("uint32")
    void_ = Keyword("void")
    widthof_ = Keyword("widthof") | Keyword("width_of")

    identifier = Word(alphas, alphanums + "_")
    as_identifier = as_ + identifier("aliasof")
    extension_identifier = of_ + identifier("aliasof")
    typeName = (boolean_ ^ char_ ^
                int16_ ^ int32_ ^
                float_ ^ float32_ ^ complex64_ ^
                uint16_ ^ uint32_ ^ int_ ^ short_)
    bufferHints = (inout_ | in_ | out_ | out_like_ | resident_)("hint")
    bufferDim = (widthof_ | heightof_ | sizeof_ | depthof_)("dim")
    paramlist = delimitedList(Group(Optional(bufferHints | bufferDim) +
                                    Optional(typeName("data_type")) +
                                    identifier("name")))
    interfaceItem = Group((Optional(kernel_ ^ void_ ^ alias_ ^ typeName ^ extension_)("kernel_type") +
                           identifier("name") +
                           Optional(as_identifier ^ extension_identifier) +
                           lparen +
                           Optional(paramlist("parameters")) + rparen + semi)
                          )

    interfaceDef = Group(interface_ +
                         identifier("name") +
                         lbrace +
                         ZeroOrMore(interfaceItem("kernel"))("kernels") +
                         rbrace + semi)
    moduleItem = interfaceDef

    bnf = OneOrMore(moduleItem)

    singleLineComment = "//" + restOfLine
    splatLineComment = "#" + restOfLine
    bnf.ignore(splatLineComment)
    bnf.ignore(singleLineComment)
    bnf.ignore(cStyleComment)

    return bnf


bnf = INTERFACECL_BNF()

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


KernelParamSpec = namedtuple("KernelParam", ["name", "hint", "dim", "pointer", "data_type", "np_dtype", "derived","derived_from"])
KernelSpec = namedtuple("KernelSpec", ["name", "alias", "parameters", "definitions"])
Interface = namedtuple("InterfaceSpec", ["name", "kernels", "aliases"])

_interfaces: Dict[str, Any] = {}


def get_interfaces(s: str) -> List[Interface]:
    """\
    Builds an InterfaceCL for the source interface definition.
    """
    try:
        interfaces = _interfaces[s]
        return interfaces
    except KeyError:
        interfaces = []
    try:
        parsed_interfaces = bnf.parseString(s)
    except ParseException as e:
        print(e.args[0], e.args[2], e.line)
        source = e.args[0].splitlines()
        cursor = " " * (e.col - 1)
        cursor += "^"
        source.insert(e.lineno, cursor)
        source = "\n".join(source)
        raise RuntimeError(f"Can't parse:\n{source}")
    for interface in parsed_interfaces:
        kernel_definitions = []
        kernel_aliases = {}
        for kernel in interface.kernels:
            aliasof = kernel.aliasof
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
            kernel_spec = KernelSpec(name=kernel_name,
                                     alias=aliasof,
                                     parameters=kernel_parameters,
                                     definitions=param_definitions)
            kernel_definitions.append(kernel_spec)
            if kernel.aliasof:
                kernel_aliases[kernel_name] = kernel.aliasof
        interfaces.append(Interface(name=interface.name,
                                    kernels=kernel_definitions,
                                    aliases=kernel_aliases))
    _interfaces[s] = interfaces
    return interfaces


def get_interface(s: str) -> Interface:
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
        alias = "self" if not kernel.alias else kernel.alias
        print(f"Kernel: {kernel.name} alias for '{alias}'", file=file)
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
                print(f"  Use as parameter of type {pointer}{param.np_dtype}", file=file)

        def returned(param, ireturn):
            print(f"  Return as {param.np_dtype} (tuple position {ireturn})", file=file)
            return ireturn + 1

        for param in kernel.parameters:
            visible_param = not param.derived
            if visible_param:
                print(f"Param {iparam}:", param, file=file)
                iparam += 1
            else:
                print(f"Param (derived):", param, file=file)
            if param.hint:
                if param.hint == "resident":
                    print(f"  Buffer is resident on GPU", file=file)
                    motion(param)
                elif param.hint in ("out_like", "outlike"):
                    print(
                        f"  Allocate an array like '{param.name}' (opencl arg position={iparam}, dtype={param.np_dtype})",
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
