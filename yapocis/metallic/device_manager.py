from __future__ import annotations

"""
DeviceManager for PyMetallic backend: coordinates a Metal device, context, queue,
and buffer manager. Also provides program build.
"""

import os
import sys
import yaml
from typing import Dict, Optional

try:
    import pymetallic as pm  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyMetallic is required for the metallic backend") from e

from yapocis.metallic.buffer_manager import BufferManager
from yapocis.metallic.device import get_device
from yapocis.utils.stereo_typing import *

YAPOCIS_CACHE = os.environ.get("YAPOCIS_CACHE", None)
if YAPOCIS_CACHE and not os.path.isdir(YAPOCIS_CACHE):
    print(f"YAPOCIS_CACHE={YAPOCIS_CACHE} is not a directory", file=sys.stderr)
    sys.exit(1)


class DeviceManager(BufferManager):
    """
    Coordinates a Metal device with a queue, a context, and a buffer manager.
    """

    def __init__(self, device_name: Optional[str | int] = None, timed: bool = False, **kwargs):
        assert device_name is None or isinstance(device_name, (str, int, bytes)), device_name
        self.device_name, self.device = get_device(device_name=device_name)
        super(DeviceManager, self).__init__(self.device, None, timed=timed)
        self.context_meta = {}

    def build(self, source: str, options=None):
        # No explicit program build step with pymetallic; kernels are compiled
        # from source on demand and cached by pymetallic.kernel_cache.
        return None

    def purge(self):
        self.purge_buffers()

    def write(self, a: Array, wait: bool = True):
        assert hasattr(a, "dtype")
        return self.write_buffer(a, wait=wait)

    def read(self, a: Array):
        return self.read_buffer(a)

    def ensure(self, a: Array):
        return self.ensure_buffer(a)


_device_managers: Dict[str | int, DeviceManager] = {}


def get_device_manager(device_name: Optional[str | int] = None) -> DeviceManager:
    try:
        dmgr = _device_managers[device_name]
    except KeyError:
        _device_managers[device_name] = dmgr = DeviceManager(device_name)
    return dmgr


def display_stats():
    device_stats = {str(dn): dmgr.stats for dn, dmgr in _device_managers.items()}
    print(yaml.safe_dump(device_stats))


def display_snapshots():
    print()
    print("Snapshots")
    for dn, dmgr in _device_managers.items():
        print(f"Device: {dn}")
        print(yaml.safe_dump(dmgr.snapshots))
