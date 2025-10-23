#!/usr/bin/env python
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.
"""
Asset management - find and list yapocis assets
"""
import sys
from argparse import ArgumentParser

from yapocis.fonts import font as find_font
from yapocis.fonts import fonts
from yapocis.icons import find_icon, icons
from yapocis.images import find_image, images
from yapocis.interfaces import find_interface, interfaces
from yapocis.kernels import find_kernel, kernels


def main():
    """Main entry point for yapocis-assets command."""
    parser = ArgumentParser()
    parser.add_argument('--images', action="store_true")
    parser.add_argument('--icons', action="store_true")
    parser.add_argument('--fonts', action="store_true")
    parser.add_argument('--kernels', action="store_true")
    parser.add_argument('--interfaces', action="store_true")
    parser.add_argument('--all', action="store_true")
    parser.add_argument('--list', action="store_true")
    parser.add_argument("assets", nargs="*")
    args = parser.parse_args()
    if args.list:
        if args.all:
            asset_types = ["interfaces","kernels","images","icons","fonts"]
        else:
            asset_types = args.assets
        for asset_type in asset_types:
            lister = globals().get(asset_type, None)
            if lister is None:
                print(f"Can't find asset manager for '{asset_type}'")
                sys.exit(1)
            print(f"## {asset_type.upper()}")
            for asset in lister():
                print(asset)
            print()
        sys.exit(0)
    if args.all:
        asset_finders = ["find_interface", "find_kernel", "find_image", "find_icon", "find_font"]
    else:
        if not args.assets:
            print("Specify asset to find")
            sys.exit(1)
        asset_finders = []
        if args.images:
            asset_finders.append("find_image")
        if args.interfaces:
            asset_finders.append("find_interface")
        if args.kernels:
            asset_finders.append("find_kernel")
        if args.icons:
            asset_finders.append("find_icon")
        if args.fonts:
            asset_finders.append("find_font")
    for asset_finder in asset_finders:
        finder = globals().get(asset_finder, None)
        if finder is None:
            print(f"Can't find asset finder '{asset_finder}'")
            sys.exit(1)
        for asset in args.assets:
            try:
                print(f"{asset_finder[5:]:10s} {asset:10s} {finder(asset)}")
            except ValueError:
                print(f"{asset_finder[5:]:10s} {asset:10s} not found")


if __name__ == '__main__':
    main()

