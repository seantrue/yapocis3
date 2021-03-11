#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.
"""
Interpret interfaces in a readable way.
"""
from argparse import ArgumentParser
from io import StringIO
from yapocis.interfaces import interfaces, interface
from yapocis.rpc.interface import get_interface, get_interfaces, load_interface, python_stubs, opencl_stubs, \
    explain_interface
from yapocis.rpc.kernel_manager import render_interface


def as_code(s):
    print()
    print("```")
    print(s.strip())
    print("```")
    print()


def as_header(s, level=0):
    print()
    print("#" * (level+2), s.strip())


def as_bullet(s, level=0):
    indent = " " * level if level else ""
    print(f"{level}* {s}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--python', action="store_true")
    parser.add_argument('--opencl', action="store_true")
    parser.add_argument('--explain', action="store_true")
    parser.add_argument('--source', action="store_true")
    parser.add_argument('--all', action="store_true")
    parser.add_argument("interfaces", nargs="*")
    args = parser.parse_args()
    requested_interfaces = [f"{name}.interface" for name in args.interfaces]
    for interface_name in interfaces():
        if requested_interfaces and interface_name not in requested_interfaces:
            continue
        as_header(interface_name.split('.')[0])
        template = load_interface(interface_name)
        source = render_interface(template)
        if "%" in source or "${" in source:
            as_bullet(f"{interface_name} has unprocessed template markup.")
            as_code(template)
            continue
        interface = get_interface(source)
        acted = 0
        if args.source or args.all:
            as_header("Interface specification",1)
            as_code(source)
        if args.python or args.all:
            acted += 1
            with StringIO() as outf:
                as_header("Python signatures",1)
                python_stubs(interface, file=outf)
                as_code(outf.getvalue())
        if args.explain or args.all:
            acted += 1
            as_header("Explanation", 1)
            with StringIO() as outf:
                explain_interface(interface, file=outf)
                as_code(outf.getvalue())
        if args.opencl or args.all:
            acted += 1
            as_header("OpenCL stubs", 1)
            with StringIO() as outf:
                opencl_stubs(interface, file = outf)
                as_code(outf.getvalue())
        if acted == 0:
            as_header("Interface specification",1)
            as_code(source)
