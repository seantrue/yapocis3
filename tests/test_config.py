#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from yapocis.ocl.device import get_device
from yapocis.utils.config import print_config


def test_print_device_config():
    name,_ = get_device()
    print("Default device")
    print_config(name)

def test_print_platform_config():
    print("All devices")
    print_config()
