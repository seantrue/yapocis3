#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from yaml import safe_dump as dump
from yapocis.rpc.engines import get_available

def get_config():
    devices = get_available()
    config = {}
    for name, device in devices.items():
        config[name] = cfg = {}
        for attr in dir(device):
            if attr.startswith("_"):
                continue
            try:
                value = getattr(device, attr)
                if callable(value):
                    continue
                if isinstance(value,(str,int,float,bool,list,tuple,dict)):
                    cfg[attr] = value
                else:
                    cfg[attr] = str(value)
            except:
                cfg[attr] = "?"
        cfg["extensions"]  = cfg.get("extensions","").split()
        cfg["platform"] = device.platform.name
    return config

def print_config():
    print(dump(get_config(),default_flow_style=None))

if __name__ == '__main__':
    print_config()
