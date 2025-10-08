#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

import yaml
from yaml.dumper import SafeDumper
import pyopencl

if hasattr(pyopencl._cl,"NameVersion"):
    def nameversion_representer(dumper, data):
      return dumper.represent_scalar('!nameversion', f'{data.name}@v:{data.version}')
    yaml.add_representer(pyopencl._cl.NameVersion, nameversion_representer,SafeDumper)
from yaml import safe_dump as dump

from yapocis.ocl.device import get_available_device_names, get_device
from yapocis.utils.stereo_typing import *

DEPRECATED_OPENCL_ATTRIBUTES = {"persistent_unique_id"}
def get_device_config(device_name:str | None=None) -> Config:
    cfg:Config = {}
    _,device = get_device(device_name)
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

def get_platform_config() -> dict[str,Config]:
    device_names  = get_available_device_names()
    config = {}
    for name in device_names:
        config[name] = get_device_config(name)
    return config

def dump_config(config):
    return dump(config,default_flow_style=None)

def print_config(name:str | None=None):
    if name is None:
        config = get_platform_config()
    else:
        config = get_device_config(name)
    print(dump_config(config))

if __name__ == '__main__':
    print_config()
