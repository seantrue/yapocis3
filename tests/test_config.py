#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from yapocis.utils.config import print_config
from yapocis.rpc.device import get_available_device_names, get_device

def test_print_device_config():
    name,_ = get_device()
    print("Default device")
    print_config(name)

def test_print_platform_config():
    print("All devices")
    print_config()
