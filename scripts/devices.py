#!/usr/bin/env python
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.
"""
List devices
"""
import sys
from argparse import ArgumentParser

from yapocis.ocl.device import get_available_device_names
from yapocis.utils.config import print_config

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--list', action="store_true")
    parser.add_argument("devices", nargs="*")
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