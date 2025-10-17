#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.

"""
Abstract base class for backend-specific device managers.

This module defines DeviceManagerBase, which establishes the common interface
for managing compute devices across all backends (OpenCL, Metal, CUDA, Numba).

All backend device managers must implement:
  - build(source, options) - Compile kernel source for device
  - __init__(device_name, timed, **kwargs) - Initialize device and context

Common functionality inherited from BufferManagerBase:
  - Buffer management (write, read, ensure, purge)
  - Statistics tracking
  - Performance timing

The DeviceManager combines:
  - Device selection and initialization
  - Context/queue/stream management
  - Buffer management (via BufferManagerBase)
  - Kernel compilation
"""

from abc import ABC, abstractmethod
from typing import Any

from yapocis.core.buffer_mgr_base import BufferManagerBase
from yapocis.utils.stereo_typing import Array


class DeviceManagerBase(BufferManagerBase, ABC):
    """
    Abstract base class for backend-specific device managers.

    Extends BufferManagerBase with device-specific operations:
      - Device selection and initialization
      - Kernel source compilation
      - High-level buffer operations (write/read/ensure/purge)

    Subclasses must implement:
        - __init__(device_name, timed, **kwargs) - Initialize device, context, and buffers
        - build(source, options) - Compile kernel source to executable program

    Attributes:
        device_name: String identifier for the device
        device: Backend-specific device object
        ctx: Backend-specific context object
        queue/stream: Backend-specific queue or stream for async operations
        context_meta: Dictionary for storing additional context metadata
    """

    def __init__(self, context: Any, queue_or_stream: Any, device_name: str, device: Any,
                 timed: bool = False, **kwargs):
        """
        Initialize device manager base.

        This is typically called by subclass __init__ after device/context creation.

        Args:
            context: Backend-specific context
            queue_or_stream: Backend-specific queue/stream
            device_name: Device identifier string
            device: Backend-specific device object
            timed: Enable performance timing
            **kwargs: Additional arguments for BufferManagerBase
        """
        super().__init__(context, queue_or_stream, timed=timed, **kwargs)
        self.device_name = device_name
        self.device = device
        self.context_meta = {}

    @abstractmethod
    def build(self, source: str, options: list[str] | None = None) -> Any:
        """
        Compile kernel source code for this device.

        Args:
            source: Kernel source code (OpenCL, CUDA, Metal, etc.)
            options: List of compiler options/flags

        Returns:
            Backend-specific compiled program/module object
        """
        pass

    def purge(self):
        """Purge unused device buffers (convenience wrapper)."""
        self.purge_buffers()

    def write(self, a: Array, wait: bool = True) -> Any:
        """
        Write numpy array to device buffer (convenience wrapper).

        Args:
            a: Numpy array to write
            wait: Whether to wait for write to complete

        Returns:
            Backend-specific buffer object
        """
        assert hasattr(a, "dtype"), "Array must be a numpy array with dtype"
        return self.write_buffer(a, wait=wait)

    def read(self, a: Array) -> Array:
        """
        Read device buffer back to numpy array (convenience wrapper).

        Args:
            a: Numpy array to read into

        Returns:
            The numpy array with data from device
        """
        return self.read_buffer(a)

    def ensure(self, a: Array) -> Any:
        """
        Ensure device buffer exists for array (convenience wrapper).

        Args:
            a: Numpy array

        Returns:
            Backend-specific buffer object
        """
        return self.ensure_buffer(a)


def get_device_manager_factory(backend_module):
    """
    Helper to create a get_device_manager() factory function.

    This provides the standard singleton pattern for device managers
    used across all backends.

    Args:
        backend_module: The backend module containing DeviceManager class

    Returns:
        Function that returns cached DeviceManager instances

    Example:
        # In yapocis/ocl/device_manager.py:
        get_device_manager = get_device_manager_factory(sys.modules[__name__])
    """
    _device_managers = {}

    def get_device_manager(device_name: str | None = None):
        """
        Get or create a DeviceManager for the specified device.

        Args:
            device_name: Device identifier (None for default)

        Returns:
            Cached DeviceManager instance
        """
        try:
            dmgr = _device_managers[device_name]
        except KeyError:
            _device_managers[device_name] = dmgr = backend_module.DeviceManager(device_name)
        return dmgr

    return get_device_manager
