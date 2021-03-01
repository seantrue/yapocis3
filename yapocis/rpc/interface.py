#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

# Process an improper subset of the CORBA IDL grammar to facilitate calling opencl from a python Numpy framework
# Inspired by an IDL parser by Paul McGuire, shipped as a demo with pyparser
#
from collections import namedtuple

from pyparsing import Optional, ParseException, OneOrMore, Literal, Keyword, Word, alphas, alphanums, \
    delimitedList, Group, ZeroOrMore, restOfLine, cStyleComment
from yapocis.utils.internals import find_yapocis_file
from yapocis.utils.typing import List

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
    boolean_ = Keyword("boolean")
    char_ = Keyword("char")
    complex64_ = Keyword("complex64")
    float_ = Keyword("float")
    float32_ = Keyword("float32")
    inout_ = Keyword("inout")
    interface_ = Keyword("interface")
    in_ = Keyword("in")
    int_ = Keyword("int")
    int16_ = Keyword("int16")
    int32_ = Keyword("int32")
    kernel_ = Keyword("kernel")
    out_ = Keyword("out")
    short_ = Keyword("short")
    uint16_ = Keyword("uint16")
    uint32_ = Keyword("uint32")
    void_ = Keyword("void")
    # Special keywords
    alias_ = Keyword("alias")
    as_ = Keyword("as")
    out_like_ = Keyword("out_like") | Keyword("outlike")
    resident_ = Keyword("resident")
    depthof_ = Keyword("depthof") | Keyword("depth_of")
    widthof_ = Keyword("widthof") | Keyword("width_of")
    heightof_ = Keyword("heightof") | Keyword("height_of")
    sizeof_ = Keyword("sizeof") | Keyword("size_of")

    identifier = Word(alphas, alphanums + "_")
    as_identifier = as_ + identifier("alias")
    typeName = (boolean_ ^ char_ ^
                int16_ ^ int32_ ^
                float_ ^ float32_ ^ complex64_ ^
                uint16_ ^ uint32_ ^ int_ ^ short_)
    bufferHints = (inout_ | in_ | out_ | out_like_ | resident_)("hint")
    bufferDim = (widthof_ | heightof_ | sizeof_ | depthof_)("dim")
    paramlist = delimitedList(Group(Optional(bufferHints | bufferDim) +
                                    Optional(typeName("data_type")) +
                                    identifier("name")))
    interfaceItem = Group((Optional(kernel_ ^ void_ ^ alias_ ^ typeName)("kernel_type") +
                           identifier("name") +
                           Optional(as_identifier("aliasof")) +
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


KernelParamSpec = namedtuple("KernelParam", ["name", "hint", "dim", "pointer", "data_type", "np_dtype", "derived"])
KernelSpec =      namedtuple("KernelSpec", ["name", "alias", "parameters", "definitions"])
Interface =       namedtuple("InterfaceSpec", ["name", "kernels", "aliases"])


def get_interfaces(s: str) -> List[Interface]:
    """\
    Builds an InterfaceCL for the source interface definition.
    """
    interfaces = []
    for interface in bnf.parseString(s):
        kernel_definitions = []
        kernel_aliases = {}
        for kernel in interface.kernels:
            alias = kernel.alias if kernel.alias else kernel.name
            param_definitions = scan_kernel_parameters(kernel)
            kernel_parameters = []
            for param in kernel.parameters:
                if param.dim:
                    data_type = "int"
                elif param.hint in ("outlike","out_like"):
                    data_type = param_definitions[param.name].data_type
                else:
                    data_type = param.data_type
                np_dtype = ctype_to_nptype.get(data_type, data_type)
                derived = True if param.dim or param.hint=="out_like" else False
                kernel_parameters.append(
                    KernelParamSpec(name=param.name,
                                    hint=param.hint,
                                    dim=param.dim,
                                    pointer=param.pointer,
                                    data_type=data_type,
                                    np_dtype=np_dtype,
                                    derived=derived))
            kernel_spec = KernelSpec(name=kernel.name,
                                     alias=alias,
                                     parameters=kernel_parameters,
                                     definitions=param_definitions)
            kernel_definitions.append(kernel_spec)
            if kernel.alias:
                kernel_aliases[kernel.name] = kernel.alias
        interfaces.append(Interface(name=interface.name,
                                    kernels=kernel_definitions,
                                    aliases=kernel_aliases))
    return interfaces

def get_interface(s: str) -> Interface:
    return get_interfaces(s)[0]

def load_interface(interface_spec:str) -> str:
    """
    Find an interface, or return source if it is an interface spec.
    Args:
        interface_spec: either an interface name or a spec

    Returns:
        an interface specification
    """
    if interface_spec.strip().startswith("interface "):
        return interface_spec
    filename= f"{interface_spec}.interface"
    path = find_yapocis_file(["interfaces", "."], filename)
    if path is None:
        raise RuntimeError(f"Can't find {filename}")
    with open(path) as inf:
        return inf.read()


def dump_interface(interface):
    print("Interface:", interface.name)
    for kernel in interface.kernels:
        print(f"Kernel: {kernel.name } alias for '{kernel.alias}")
        iparam = 0
        for param in kernel.parameters:
            to_gpu = param.hint in ("in", "inout")
            from_gpu = param.hint in ("out", "inout", "out_like")
            on_gpu = param.hint in ("resident","working")
            visible_param = not param.derived
            if visible_param:
                print(f"Param {iparam}:", param)
                iparam += 1
            else:
                print(f"Param (derived):", param)
            if param.hint:
                if param.hint == "resident":
                    print(f"  Buffer is resident on GPU")
                elif param.hint == "out_like":
                    print(f"  Allocate a buffer like '{param.name}' position={iparam}")
                else:
                    print(f"  Coerce '{param.name}' to {param.np_dtype}")
                    print(f"  Allocate a buffer for {param.name}")
            if param.dim:
                print(f"  Pass in {param.dim} of `{param.name}`")
            if to_gpu:
                print("  Copy to GPU")
            if from_gpu:
                print("  Copy back from GPU")
            if on_gpu:
                print("  Already on GPU")
            if not from_gpu:
                print("  Use as parameter")
            print()
        print()

def opencl_stubs(interface):
    print("// kernels", interface.name)
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
        print(f"__kernel void {kernel.name}({', '.join(kernel_parameters)}) {{")
        print("}")
    print()


def python_stubs(interface):
    print("## interface", interface.name)
    kernels = interface.kernels
    for kernel in kernels:
        python_parameters = []
        python_returns = []
        for parameter in kernel.parameters:
            if parameter.hint in ("out_like", "out"):
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
        print(f"def {kernel.name}({', '.join(python_parameters)}) -> {python_returns}:")
        print("    pass")
    print()
