#!/usr/bin/env python
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.
"""
Interpret interfaces in a readable way - generate documentation and stubs
"""
from argparse import ArgumentParser
from io import StringIO

from yapocis.core.interface import load_interface, scan_kernel_parameters, get_interface
from yapocis.core.manager import get_manager
from yapocis.interfaces import interfaces


def explain_interface(interface, file=None):
    print("Interface:", interface.name, file=file)
    for kernel in interface.kernels:
        tiling = "tiled" if kernel.tiled else "untiled"
        if kernel.alias:
            print(
                f"Kernel: {kernel.name} {tiling} alias for '{kernel.alias}'", file=file
            )
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
                print(
                    f"  Pass as parameter of type {pointer}{param.np_dtype}", file=file
                )

        def returned(param, ireturn):
            print(f"  Return as {param.np_dtype} (tuple position {ireturn})", file=file)
            return ireturn + 1

        def format_param(param):
            return f"{param.data_type} {param.pointer + ' ' if param.pointer else ''}{param.name}"

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
                        file=file,
                    )
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
                kernel_parameters.append(
                    f"__global {parameter.data_type} *out_like_{parameter.name}"
                )
            elif parameter.hint:
                kernel_parameters.append(
                    f"__global {parameter.data_type} *{parameter.name}"
                )
            elif parameter.dim:
                kernel_parameters.append(f"int {parameter.dim}_{parameter.name}")
            else:
                kernel_parameters.append(f"{parameter.data_type} {parameter.name}")
        print(
            f"__kernel void {kernel.name}({', '.join(kernel_parameters)}) {{", file=file
        )
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
        print(
            f"def {kernel.name}({', '.join(python_parameters)}) -> {python_returns}:",
            file=file,
        )
        print("    pass", file=file)
    print(file=file)


def as_code(s):
    print()
    print("```")
    print(s.strip())
    print("```")
    print()


def as_header(s, level=0):
    print()
    print("#" * (level + 2), s.strip())


def as_bullet(s, level=0):
    indent = (" " * level) if level else ""
    print(f"{indent}* {s}")


def main():
    """Main entry point for yapocis-interface-spec command."""
    parser = ArgumentParser()
    parser.add_argument("--python", action="store_true")
    parser.add_argument("--opencl", action="store_true")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--source", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("interfaces", nargs="*")
    args = parser.parse_args()
    mgr = get_manager()
    requested_interfaces = [f"{name}.interface" for name in args.interfaces]
    if not args.list:
        print("# Kernels")
    for interface_name in interfaces():
        if requested_interfaces and interface_name not in requested_interfaces:
            continue
        if args.list:
            print(interface_name)
            continue
        s = interface_name.split(".")[0]
        as_header(f"`{s}`")
        template = load_interface(interface_name)
        source = mgr.render_interface(template)
        if "%" in source or "${" in source:
            as_bullet(f"{interface_name} has unprocessed template markup.")
            as_code(template)
            continue
        try:
            interface = get_interface(source)
        except RuntimeError as e:
            print(e)
            print(source)
            continue
        acted = 0
        if args.python or args.all:
            acted += 1
            with StringIO() as outf:
                as_header("Signatures", 1)
                python_stubs(interface, file=outf)
                as_code(outf.getvalue())
        if args.source or args.all:
            as_header("Specification", 1)
            as_code(source)
        if args.explain or args.all:
            acted += 1
            as_header("Explanation", 1)
            with StringIO() as outf:
                explain_interface(interface, file=outf)
                as_code(outf.getvalue())
        if args.opencl or args.all:
            acted += 1
            as_header("Stubs", 1)
            with StringIO() as outf:
                opencl_stubs(interface, file=outf)
                as_code(outf.getvalue())
        if acted == 0:
            as_header("Interface specification", 1)
            as_code(source)


if __name__ == "__main__":
    main()
