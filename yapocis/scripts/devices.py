
#!/usr/bin/env python
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2025.
"""
list OpenCL devices and their configurations
"""
import sys
from argparse import ArgumentParser

from yapocis.ocl.device import get_available_device_names
from yapocis.utils.config import print_config

def main():
    """Main entry point for yapocis-devices command."""
    parser = ArgumentParser(description="list OpenCL devices and configurations")
    parser.add_argument('--list', action="store_true", help="list device names only")
    parser.add_argument("devices", nargs="*", help="Specific devices to query")
    args = parser.parse_args()

    device_names = get_available_device_names()
    if args.list:
        for device_name in device_names:
            print(device_name)
        sys.exit(0)

    if args.devices:
        device_names = list(set(args.devices) & set(device_names))

    for device_name in device_names:
        print(device_name)
        print_config(device_name)

if __name__ == '__main__':
    main()