from __future__ import annotations

"""
DeviceManager for PyMetallic backend: coordinates a Metal device, context, queue,
and buffer manager. Also provides program build.
"""

import os
import sys
import yaml
import numpy as np

try:
    import pymetallic  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyMetallic is required for the metallic backend") from e

from yapocis.core.device_mgr_base import DeviceManagerBase
from yapocis.metallic.device import get_device
from yapocis.utils.stereo_typing import Array

YAPOCIS_CACHE = os.environ.get("YAPOCIS_CACHE", None)
if YAPOCIS_CACHE and not os.path.isdir(YAPOCIS_CACHE):
    print(f"YAPOCIS_CACHE={YAPOCIS_CACHE} is not a directory", file=sys.stderr)
    sys.exit(1)


class DeviceManager(DeviceManagerBase):
    """
    Metal device manager - coordinates device, queue, and buffer management.

    Inherits from DeviceManagerBase (which includes buffer management functionality).
    """

    def __init__(
        self, device_name: str | int | None = None, timed: bool = False, **kwargs
    ):
        """
        Initialize Metal device manager.

        Args:
            device_name: Device identifier (None for default)
            timed: Enable performance timing
            **kwargs: Additional arguments for DeviceManagerBase
        """
        assert device_name is None or isinstance(
            device_name, (str, int, bytes)
        ), device_name
        device_name_str, device = get_device(device_name=device_name)
        queue = device.make_command_queue()
        # Metal doesn't have a separate context object; use device as context
        super().__init__(device, queue, device_name_str, device, timed=timed, **kwargs)

    def build(self, source: str, options: list[str] | None = None):
        """
        Compile Metal source code.

        Metal kernels are compiled on demand and cached by pymetallic,
        so this is a no-op that returns None.

        Args:
            source: Metal kernel source code
            options: Compiler options (unused for Metal)

        Returns:
            None
        """
        # No explicit program build step with pymetallic; kernels are compiled
        # from source on demand and cached by pymetallic.kernel_cache.
        return None

    # Implement abstract methods from DeviceManagerBase
    def make_buffer(self, a: Array) -> "pymetallic.Buffer":
        """Create PyMetallic buffer from numpy array."""
        # Ensure contiguous data for low-level transfer
        if not a.flags.c_contiguous:
            a = np.ascontiguousarray(a)
        buf = pymetallic.Buffer.from_numpy(self.device, a)
        buffer_id = id(a)
        self.arrays[buffer_id] = a
        self.buffers[buffer_id] = buf
        return buf

    def read_buffer(self, a: Array) -> Array:
        """Read PyMetallic buffer back to numpy array."""
        buf = self.find_buffer(a, self.READ)
        view = buf.to_numpy(dtype=a.dtype, shape=a.shape)
        np.copyto(a, view)
        return a

    def write_buffer(self, a: Array, wait: bool = True) -> "pymetallic.Buffer":
        """Write numpy array to PyMetallic buffer."""
        # For PyMetallic we create/replace the backing buffer from host bytes
        if not a.flags.c_contiguous:
            a = np.ascontiguousarray(a)
        buf = pymetallic.Buffer.from_numpy(self.device, a)
        buffer_id = id(a)
        self.arrays[buffer_id] = a
        self.buffers[buffer_id] = buf
        return buf

    def find_buffer(self, a: Array, op: int) -> "pymetallic.Buffer | None":
        """Find existing PyMetallic buffer for array."""
        assert op in (
            self.READ,
            self.WRITE,
        ), f"{op=} not one of {self.READ=},{self.WRITE=}"
        buffer_id = id(a)
        havea = buffer_id in self.buffers and buffer_id in self.arrays
        if havea:
            self.hits += 1
            return self.buffers[buffer_id]
        else:
            self.misses += 1
        if not havea:
            if op == self.READ:
                raise ValueError(
                    f"Array {buffer_id=} not in yapocis management, you may have written to it, or be using an assigned or .copy"
                )
            return None
        raise RuntimeError(f"Failed to find a buffer for {op} a@{buffer_id=}")

    def wait_events(self, events: list) -> None:
        """
        Wait for a list of Metal command buffers to complete.

        Args:
            events: List of command buffers or events to wait for
        """
        if events:
            # Metal command buffers have waitUntilCompleted() method
            for event in events:
                if hasattr(event, 'waitUntilCompleted'):
                    event.waitUntilCompleted()

_device_managers: dict[str | int, DeviceManager] = {}

def get_device_manager(device_name: str | int | None = None) -> DeviceManager:
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
