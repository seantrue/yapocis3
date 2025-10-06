#!/usr/bin/env python3
"""Check how the interface is parsed."""
from yapocis.core.interface import load_interface, get_interface

iface_txt = load_interface('imageops')
print("=== Interface Text ===")
print(iface_txt)
print()

iface = get_interface(iface_txt)
print("=== Parsed Interface ===")
print(f"Name: {iface.name}")
print(f"Kernels: {[k.name for k in iface.kernels]}")
print()

for k in iface.kernels:
    print(f"Kernel: {k.name}")
    if hasattr(k, 'alias') and k.alias:
        print(f"  Alias: {k.alias}")
    if hasattr(k, 'args'):
        print(f"  Args ({len(k.args)}):")
        for i, arg in enumerate(k.args):
            print(f"    [{i}] {arg}")
            print(f"        name: {getattr(arg, 'name', 'N/A')}")
            print(f"        qualifier: {getattr(arg, 'qualifier', 'N/A')}")
            print(f"        type: {getattr(arg, 'type', 'N/A')}")
            print(f"        like: {getattr(arg, 'like', None)}")
    else:
        print(f"  No args attribute")
    print()
