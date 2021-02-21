#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

# Process an improper subset of the CORBA IDL grammar to facilitate calling opencl from a python Numpy framework
# Inspired by an IDL parser by Paul McGuire, shipped as a demo with pyparser
#
from collections import namedtuple
from pyparsing import ParseException

from yapocis.rpc.parser import INTERFACECL_BNF, expected
from yapocis.utils.basic import find_yapocis_file
from yapocis.utils.typing import *


class InterfaceCL:
    """
    Manages the kernel definitions, particularly the parameter specifications.
    """

    def __init__(self, interface_name: str, kernel_defs: Dict, kernel_aliases: Dict[str, str]):
        self.interface_name = interface_name
        self.kernel_defs = kernel_defs
        self.kernel_aliases = kernel_aliases

    def kernels(self):
        "Returns a list of kernel names"
        return list(self.kernel_defs.keys())

    def kernel_params(self, kernel):
        "Returns the parameter specifications for a kernel"
        return self.kernel_defs.get(kernel, None)

    def kernel_alias(self, kernel_name):
        return self.kernel_aliases.get(kernel_name, kernel_name)

    def __str__(self):
        return self.interface_name
# Map from OpenCL datatype to a numpy data type

cl2np = {"char":   "int8",
         "uchar":  "uint8",
         "short":  "int16",
         "ushort": "uint16",
         "int":    "int32",
         "uint":   "uint32",
         "long":   "int64",
         "ulong":  "uint64",
         "half":   "float16",
         "float":  "float32",
         "double": "float64"}
np2cl = {value:key for key,value in cl2np.items()}

ArgSpec = namedtuple("ArgSpec",["buffer_hint", "direction", "dtype", "name"])
def fix_param(param: List[Any]) -> List[Any]:
    """\
    Deals with variants of parameter specifications, returns a uniform array of information.
    """
    # Length 2: outlike, identifier (exists)
    # Length 3: outlike, dtype, identifier (exists)
    # Length 3: in, dtype, identifier 
    # Length 4 direction, dtype, *, identifier
    op = ["", "", "", ""]
    if len(param) == 2:
        expected(param[0], ("outlike", "resident", "sizeof", "widthof", "heightof", "depthof"))
        op[0] = param[0]
        op[2] = '*'
        op[3] = param[1]
    elif len(param) == 3:
        expected(param[0], ("in", "outlike", "resident", "sizeof", "widthof", "heightof", "depthof"))
        if param[0] in ("outlike", "sizeof", "widthof", "heightof", "depthof"):
            op[0] = param[0]
            op[1] = param[1]
            op[2] = '*'
            op[3] = param[2]
        else:
            op[0] = param[0]
            op[1] = param[1]
            op[3] = param[2]
    else:
        op = param
    op[1] = cl2np.get(op[1], op[1])
    return op

class ArgSpecError(Exception):
    pass

ArgSpec = namedtuple("ArgSpec",["buffer_hint", "direction", "dtype", "name"])


def get_interface(s: str) -> InterfaceCL:
    """
    Builds an InterfaceCL for the source interface definition.
    """
    try:
        bnf = INTERFACECL_BNF
        tokens = bnf.parseString(s)
        tokens = tokens.asList()
        tokens = tokens.pop(0)
        assert tokens.pop(0) == "interface"
        interface_name = tokens.pop(0)
        kernel_defs = {}
        kernel_aliases = {}
        assert tokens.pop(0) == "{"
        while 1:
            token = tokens.pop(0)
            if token == "}":
                break
            assert token in ("kernel", "alias")
            if token == "alias":
                alias = tokens.pop(0)
                assert tokens[0][0] == "as"
                kernel_name = tokens.pop(0)[1]
                kernel_aliases[alias] = kernel_name
                kernel_name = alias
            else:
                alias = None
                kernel_name = tokens.pop(0)
            params = []
            assert tokens.pop(0) == '('
            while 1:
                param = tokens.pop(0)
                if param == ")":
                    break
                params.append(fix_param(param))
            assert tokens.pop(0) == ";"
            kernel_defs[kernel_name] = params
        return InterfaceCL(interface_name, kernel_defs, kernel_aliases)
    except ParseException as err:
        print(err.line)
        print(" " * (err.column - 1) + "^")
        print(err)
        raise


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
    filename = f"{interface_spec}.interface"
    path = find_yapocis_file(["interfaces", "."], filename)
    if path is None:
        raise RuntimeError(f"Can't find {filename}")
    with open(path) as inf:
        return inf.read()