#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.

import os
from typing import Dict, Optional, List

import pycuda.driver as cuda


def get_available_device_names():
    """Get names of all available CUDA devices"""
    cuda.init()
    device_names = []
    for i in range(cuda.Device.count()):
        device = cuda.Device(i)
        device_names.append(device.name())
    return device_names


def get_available_devices(aliases: bool = True) -> Dict[str, cuda.Device]:
    """
    Get the names of the available CUDA devices.

    Returns: a dictionary mapping names to devices, suitable for use with get_device

    """
    cuda.init()
    devices = {}
    for i in range(cuda.Device.count()):
        device = cuda.Device(i)
        name = device.name()
        devices[name] = device
        if aliases:
            devices[i] = device
    return devices


def pick_default_device(device_names: List[str]) -> str:
    """Pick the default CUDA device - prefer first GPU found"""
    if not device_names:
        raise RuntimeError("No CUDA devices found")
    # Just pick the first device for CUDA
    return device_names[0]


def get_device(device_name: Optional[str] = None) -> tuple[str, cuda.Device]:
    """
    Get a device that is an exact or approximate match for device name, which can also be a numeric index,
    and if None, will be the first device found.

    Args:
        device_name: if present, either a name or string/int that can be index into the device dictionary

    Returns: tuple of (device_name, pycuda.driver.Device)

    """
    device_name = (
        os.environ.get("YAPOCIS_ENGINE", None) if device_name is None else device_name
    )
    devices = get_available_devices()
    if not devices:
        raise RuntimeError("No CUDA devices found")
    device_names = [name for name in devices.keys() if isinstance(name, str)]
    if not device_name:
        device_name = pick_default_device(device_names)
        print(f"Defaulting to {device_name}")
    if device_name not in devices:
        try:
            engine_index = int(device_name)
        except ValueError:
            raise RuntimeError(
                f"{device_name} not an available device or device index  {device_names}"
            )
        except TypeError:
            raise RuntimeError(f"{device_name} not a string or integer")
        return get_device(device_names[engine_index])
    return device_name, devices[device_name]
