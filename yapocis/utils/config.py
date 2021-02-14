#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from yaml import safe_dump as dump
from yapocis.rpc.engines import get_available, get_yapocis_device
from yapocis.utils.typing import *

DEPRECATED_OPENCL_ATTRIBUTES = {"persistent_unique_id"}
def get_device_config(device_name:Optional[str]=None) -> Config:
    cfg:Config = {}
    _,device = get_yapocis_device(device_name)
    for attr in dir(device):
        if attr.startswith("_"):
            continue
        if attr in DEPRECATED_OPENCL_ATTRIBUTES:
            continue
        try:
            value = getattr(device, attr)
            if callable(value):
                continue
            if isinstance(value, (str, int, float, bool, list, tuple, dict)):
                cfg[attr] = value
            else:
                cfg[attr] = str(value)
        except:
            cfg[attr] = "?"
    cfg["extensions"] = cfg.get("extensions", "").split()
    cfg["__platform__"] = device.platform.name
    cfg["__name__"] = cfg["name"]
    return cfg

def get_platform_config() -> Dict[str,Config]:
    device_names  = list(get_available().keys())
    config = {}
    for name in device_names:
        config[name] = get_device_config(name)
    return config

def dump_config(config):
    return dump(config,default_flow_style=None)

def print_config(name:Optional[str]=None):
    if name is None:
        config = get_platform_config()
    else:
        config = get_device_config(name)
    print(dump_config(config))

if __name__ == '__main__':
    print_config()
