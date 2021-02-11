#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from yapocis.utils.config import get_platform_config, get_yapocis_device, get_device_config, print_config


def test_print_device_config():
    name,_ = get_yapocis_device()
    print("Default device")
    print_config(name)

def test_print_platform_config():
    print("All devices")
    print_config()
