from __future__ import annotations

"""
Device enumeration and selection for PyMetallic (Apple Metal) backend.

Assumes a PyOpenCL-like API surface provided by `pymetallic`:
  - `get_devices()` -> list[Device]
  - Each device has a `.name` attribute
"""

import os

try:
    import pymetallic as pm  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyMetallic is required for the metallic backend") from e


def get_available_device_names() -> list[str]:
    devices = pm.Device.get_all_devices()
    return [getattr(d, "name", str(d)) for d in devices]


def get_available_devices(aliases: bool = True) -> dict[str | int, pm.Device]:
    """
    Build a mapping of device identifiers to Device objects.

    Keys include the device name (string) and, if `aliases` is True, numeric indices.
    """
    devices = pm.Device.get_all_devices()
    out: dict[str | int, pm.Device] = {}
    for i, dev in enumerate(devices):
        name = getattr(dev, "name", None) or f"metal:{i}"
        out[name] = dev
        if aliases:
            out[i] = dev
    return out


def pick_default_device(device_names: list[str | int]) -> str | int:
    # Prefer discrete GPU names if present; otherwise pick the last item.
    lowercase = {str(n).lower(): n for n in device_names if isinstance(n, (str, int))}
    preferences = ("radeon", "amd", "apple", "gpu")
    for pref in preferences:
        for dn in lowercase.keys():
            if pref in dn:
                return lowercase[dn]
    return device_names[-1]


def get_device(
    device_name: str | int | None = None,
) -> tuple[str | int, pm.Device]:
    """
    Resolve a Metal device by name or index. If not provided, uses `YAPOCIS_DEVICE`
    or chooses a default via `pick_default_device`.
    """
    device_name = (
        os.environ.get("YAPOCIS_DEVICE", None) if device_name is None else device_name
    )
    devices = get_available_devices()
    if not devices:
        raise RuntimeError("No Metal devices found via PyMetallic")
    keys = list(devices.keys())
    if not device_name:
        device_name = pick_default_device(keys)
        print(f"Defaulting to Metal device {device_name}")
    if device_name not in devices:
        try:
            idx = int(device_name)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            raise RuntimeError(
                f"{device_name} not an available Metal device or index {keys}"
            )
        return get_device(keys[idx])
    return device_name, devices[device_name]
