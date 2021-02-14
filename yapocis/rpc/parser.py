#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

#
# interfacecl.py
#
# Process an improper subset of the CORBA IDL grammar to facilitate calling opencl from a python Numpy framework
# Inspired by an IDL parser by Paul McGuire, shipped as a demo with pyparser
#
from yapocis.utils.typing import *
from pyparsing import Literal, Word, OneOrMore, ZeroOrMore, \
    delimitedList, Group, Optional, alphas, restOfLine, cStyleComment, \
    alphanums, ParseException, Keyword

bnf = None


def INTERFACECL_BNF() -> OneOrMore:
    """\
    pyparser grammar for the yapocis interface specification. Inspired by an IDL parser by Paul McGuire, shipped as a demo with pyparser.
    """
    global bnf

    if not bnf:
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
        outlike_ = Keyword("outlike")
        resident_ = Keyword("resident")
        widthof_ = Keyword("widthof")
        heightof_ = Keyword("heightof")
        sizeof_ = Keyword("sizeof")

        identifier = Word(alphas, alphanums + "_")
        typeName = (
                    boolean_ ^ char_ ^ int16_ ^ int32_ ^ float32_ ^ complex64_ ^ uint16_ ^ uint32_ ^ int_ ^ float_ ^ short_)
        bufferHints = (inout_ | in_ | out_ | outlike_ | resident_ | widthof_ | heightof_ | sizeof_)
        paramlist = delimitedList(Group(bufferHints + Optional(typeName) + Optional(star) + identifier))
        interfaceItem = ((kernel_ ^ void_ ^ alias_ ^ typeName) + identifier + Optional(
            Group(as_ + identifier)) + lparen + Optional(paramlist) + rparen + semi)
        interfaceDef = Group(interface_ + identifier + lbrace + ZeroOrMore(interfaceItem) + rbrace + semi)
        moduleItem = interfaceDef

        bnf = OneOrMore(moduleItem)

        singleLineComment = "//" + restOfLine
        splatLineComment = "#" + restOfLine
        bnf.ignore(splatLineComment)
        bnf.ignore(singleLineComment)
        bnf.ignore(cStyleComment)

    return bnf


class InterfaceCL:
    """
    Manages the kernel definitions, particularly the parameter specifications.
    """

    def __init__(self, interface_name: str, kernel_defs: List[Any], kernel_aliases: Dict[str, str]):
        self.interface_name = interface_name
        self.kernel_defs = kernel_defs
        self.kernel_aliases = kernel_aliases
        self.program = None

    def kernels(self):
        "Returns a list of kernel names"
        return list(self.kernel_defs.keys())

    def kernel_params(self, kernel):
        "Returns the parameter specifications for a kernel"
        return self.kernel_defs.get(kernel, None)

    def kernel_alias(self, kernel_name):
        return self.kernel_aliases.get(kernel_name, kernel_name)


dtypemap = {"int": "int32", "float": "float32", "short": "int16"}


def expected(token:str, tokens:List[str]):
    if token in tokens:
        return
    raise ValueError("Found %s, expected %s" % (token, tokens))


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
        expected(param[0], ("outlike", "resident", "sizeof", "widthof", "heightof"))
        op[0] = param[0]
        op[2] = '*'
        op[3] = param[1]
    elif len(param) == 3:
        expected(param[0], ("in", "outlike", "resident", "sizeof", "widthof", "heightof"))
        if param[0] in ("outlike", "sizeof", "widthof", "heightof"):
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
    # Map from OpenCL datatype to a numpy data type
    op[1] = dtypemap.get(op[1], op[1])
    return op


def get_interface(s: str) -> InterfaceCL:
    """\
    Builds an InterfaceCL for the source interface definition.
    """
    try:
        bnf = INTERFACECL_BNF()
        tokens = bnf.parseString(s)
        tokens = tokens.asList()
        tokens = tokens.pop(0)
        assert tokens.pop(0) == "interface"
        interfacename = tokens.pop(0)
        kerneldefs = {}
        kernelaliases = {}
        assert tokens.pop(0) == "{"
        while 1:
            token = tokens.pop(0)
            if token == "}":
                break
            assert token in ("kernel", "alias")
            if token == "alias":
                alias = tokens.pop(0)
                assert tokens[0][0] == "as"
                kernelname = tokens.pop(0)[1]
                kernelaliases[alias] = kernelname
                kernelname = alias
            else:
                alias = None
                kernelname = tokens.pop(0)
            params = []
            assert tokens.pop(0) == '('
            while 1:
                param = tokens.pop(0)
                if param == ")":
                    break
                params.append(fix_param(param))
            assert tokens.pop(0) == ";"
            kerneldefs[kernelname] = params
        return InterfaceCL(interfacename, kerneldefs, kernelaliases)
    except ParseException as err:
        print(err.line)
        print(" " * (err.column - 1) + "^")
        print(err)
        raise
