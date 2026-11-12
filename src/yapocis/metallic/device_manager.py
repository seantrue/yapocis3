from __future__ import annotations

"""
DeviceManager for PyMetallic backend: coordinates a Metal device, context, queue,
and buffer manager. Also provides program build.
"""

import os
import sys

import numpy as np
import yaml

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
    def make_buffer(self, a: Array) -> pymetallic.Buffer:
        """Create PyMetallic buffer from numpy array."""
        # Register under the caller's array, whatever its layout (#89).
        # Buffer.from_numpy makes its own C-contiguous copy and Metal copies
        # the bytes, so no copy has to be kept alive here.
        buf = pymetallic.Buffer.from_numpy(self.device, a)
        self._register(a, buf)
        return buf

    def make_output_buffer(self, a: Array) -> pymetallic.Buffer:
        """Allocate a buffer for an output array without copying its contents.

        For out_like results only: the host array is a fresh ``np.empty_like``
        whose bytes are garbage, so ``make_buffer``'s host-to-device copy is
        wasted (12x slower at 512x512 float32, 400x at 2048x2048x4). Arrays
        whose contents matter (in, inout, resident) must keep using
        ``make_buffer``.
        """
        if a.nbytes == 0:
            return self.make_buffer(a)
        buf = pymetallic.Buffer(self.device, a.nbytes)
        self._register(a, buf)
        return buf

    def read_buffer(self, a: Array) -> Array:
        """Read PyMetallic buffer back to numpy array."""
        buf = self.find_buffer(a, self.READ)
        view = buf.to_numpy(dtype=a.dtype, shape=a.shape)
        np.copyto(a, view)
        return a

    def write_buffer(self, a: Array, wait: bool = True) -> pymetallic.Buffer:
        """
        Write numpy array to PyMetallic buffer.

        Args:
            a: NumPy array to write to GPU
            wait: If True, wait for transfer to complete (sync).
                  If False, return immediately (async transfer).

        Returns:
            PyMetallic buffer handle

        Note:
            On Apple Silicon with unified memory, buffer "transfers" are
            essentially pointer operations, so async_write provides no benefit.
            The wait parameter is accepted for API compatibility but currently
            ignored - all writes use the fast synchronous Buffer.from_numpy().
        """
        # Register under the caller's array, as make_buffer does (#89).

        # Always use synchronous Buffer.from_numpy on Metal
        # Reasons (rechecked on pymetallic 0.3.1, M4 Max, #90):
        # 1. Unified memory: from_numpy is one copy into a shared buffer
        # 2. async_buffer_from_numpy needs a command buffer, a commit and a wait,
        #    and writes private storage, which read_buffer can't map
        # 3. Command buffer overhead (~150μs) >> a small transfer (~6μs at 4 KiB)
        # 4. Result: async is ~1.7-26x SLOWER than sync, even when 8 writes
        #    share one command buffer
        buf = pymetallic.Buffer.from_numpy(self.device, a)
        # A new buffer per write; the store keeps one weak ref per live array and
        # drops the previous buffer.
        self._register(a, buf)
        return buf

    def wait_events(self, events: list) -> None:
        """
        Wait for a list of Metal command buffers to complete.

        Args:
            events: List of command buffers or events to wait for
        """
        if events:
            # Metal command buffers have waitUntilCompleted() method
            for event in events:
                if hasattr(event, "waitUntilCompleted"):
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
