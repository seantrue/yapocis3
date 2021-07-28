#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import os
from typing import Dict, Optional

from pyopencl._cl import get_platforms


def get_available_device_names():
    platforms = get_platforms()
    device_names = []
    for platform in platforms:
        for device in platform.get_devices():
            if device.name not in device_names:
                device_names.append(device.name)
            else:
                device_names.append(f"{platform.name}:{device.name}")
    return device_names


def get_available_devices(aliases:bool=True) -> Dict[str, "Device"]:
    """
    Get the names of the available OpenCL platforms and devices. Name is device name,
    with an alias of platform:device name.

    Returns: a dictionary mapping names to devices, suitable for use with get_yapocis_device

    """
    platforms = get_platforms()
    devices = {}
    i = 0
    for platform in platforms:
        for device in platform.get_devices():
            if device.name not in devices:
                devices[device.name] = device
            else:
                devices[f"{platform.name}:{device.name}"] = device
            if aliases:
                devices[i] = device
            i += 1
    return devices


def get_device(device_name: Optional[str] = None) -> "Device":
    """
    Get a device that is an exact or approximate match for device name, which can also be an numeric index,
    and if None, will be the first device found.

    Args:
        device_name: if present, either a name or string/int that can be index into the device dictionary

    Returns: an OpenCL device as wrapped by PyOpenCL

    """
    device_name = os.environ.get("YAPOCIS_ENGINE", None) if device_name is None else device_name
    devices = get_available_devices()
    if not devices:
        raise RuntimeError("No OpenCL devices found")
    device_names = list(devices.keys())
    if device_name is None:
        device_name = device_names[0]
        print(f"Defaulting to {device_name}")
    if device_name not in devices:
        try:
            engine_index = int(device_name)
        except ValueError:
            raise RuntimeError(f"{device_name} not an available device {device_names}")
        return get_device(device_names[engine_index])
    return device_name, devices[device_name]