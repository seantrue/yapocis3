#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.

"""
Abstract base class for backend-specific device managers.

This module defines DeviceManagerBase, which establishes the common interface
for managing compute devices and their buffers across all backends (OpenCL, Metal, CUDA, Numba).

All backend device managers must implement:
  - build(source, options) - Compile kernel source for device
  - __init__(device_name, timed, **kwargs) - Initialize device and context
  - make_buffer(a) - Create a new device buffer from numpy array
  - read_buffer(a) - Read device buffer back to numpy array
  - write_buffer(a, wait) - Write numpy array to device buffer
  - find_buffer(a, op) - Find existing buffer for array

Common functionality provided by base class:
  - Buffer management (write, read, ensure, purge)
  - Statistics tracking (hits, misses, purged buffers)
  - Performance timing (optional)
  - Snapshot/stats reporting

The DeviceManager combines:
  - Device selection and initialization
  - Context/queue/stream management
  - Buffer management with weak reference tracking
  - Kernel compilation
"""

import time
import warnings
from abc import ABC, abstractmethod
from typing import Any
from weakref import WeakValueDictionary

from yapocis.utils.stereo_typing import Array


def method_timer(self, name, method):
    """Decorator to time method execution."""

    def timed_method(*args, **kwargs):
        t = time.time()
        value = method(*args, **kwargs)
        t = time.time() - t
        self.times[name] += t
        return value

    return timed_method


class DeviceManagerBase(ABC):
    """
    Abstract base class for backend-specific device managers.

    Manages both device operations and buffer lifecycle, tracking usage statistics
    and providing common buffer management operations.

    Class constants:
        READ: Operation type for reading from device
        WRITE: Operation type for writing to device
        DEBUG: Enable debug output
        TIMED: Enable performance timing
        TIMED_METHODS: Methods to time when TIMED=True

    Subclasses must implement:
        - __init__(device_name, timed, **kwargs) - Initialize device, context, and buffers
        - build(source, options) - Compile kernel source to executable program
        - make_buffer(a) -> backend-specific buffer type
        - read_buffer(a) -> Array
        - write_buffer(a, wait=True) -> backend-specific buffer type
        - find_buffer(a, op) -> backend-specific buffer type | None

    Attributes:
        device_name: String identifier for the device
        device: Backend-specific device object
        ctx: Backend-specific context object
        queue/stream: Backend-specific queue or stream for async operations
        context_meta: Dictionary for storing additional context metadata
        arrays: WeakValueDictionary of numpy array references
        buffers: Dictionary of device buffers (strong references)
        hits/misses: Buffer cache statistics
        purged: Count of purged buffers
    """

    READ = 0
    WRITE = 1
    DEBUG = True
    TIMED = False
    TIMED_METHODS = ("make_buffer", "ensure_buffer", "read_buffer", "write_buffer")

    # Resource usage thresholds
    BUFFER_WARNING_THRESHOLD = 100  # Warn when buffer count exceeds this
    LEAKED_BUFFER_WARNING_THRESHOLD = 50  # Warn when leaked buffers exceed this

    def __init__(
        self,
        context: Any,
        queue_or_stream: Any,
        device_name: str,
        device: Any,
        timed: bool = False,
        autoflatten: bool = True,
        **kwargs,
    ):
        """
        Initialize device manager base.

        This is typically called by subclass __init__ after device/context creation.

        Args:
            context: Backend-specific context
            queue_or_stream: Backend-specific queue/stream
            device_name: Device identifier string
            device: Backend-specific device object
            timed: Enable performance timing
            autoflatten: Automatically flatten arrays for buffer operations
            **kwargs: Additional arguments (reserved for future use)
        """
        self.ctx = context
        self.queue = queue_or_stream  # Generic name for queue/stream
        self.device_name = device_name
        self.device = device
        self.autoflatten = autoflatten
        self.context_meta = {}

        # Buffer tracking
        self.arrays = WeakValueDictionary()  # Weak references to numpy arrays
        self.buffers = {}  # Strong references to device buffers

        # Statistics
        self.hits = self.misses = 0
        self.purged = 0
        self.snapshots = []

        # Performance timing
        self.timed = timed
        self.times = dict.fromkeys(self.TIMED_METHODS, 0)
        if self.timed or self.TIMED:
            for name in self.TIMED_METHODS:
                setattr(self, name, method_timer(self, name, getattr(self, name)))

    def stats(self, **kwargs) -> dict:
        """
        Get current buffer manager statistics.

        Returns:
            dict: Statistics including hits, misses, purged count, buffer count, and timings
        """
        leaked = self.count_leaked_buffers()
        d = dict(
            purged=self.purged,
            hits=self.hits,
            misses=self.misses,
            buffers=len(self.buffers),
            arrays=len(self.arrays),
            leaked=leaked,
        )
        d.update(self.times)
        d.update(kwargs)
        return d

    def count_leaked_buffers(self) -> int:
        """
        Count buffers that no longer have corresponding array references.

        Returns:
            Number of leaked buffers (buffers without array references)
        """
        leaked = 0
        for buffer_id in self.buffers.keys():
            if buffer_id not in self.arrays:
                leaked += 1
        return leaked

    def check_resource_usage(self, warn: bool = True) -> dict:
        """
        Check current resource usage and optionally warn if thresholds exceeded.

        Args:
            warn: Whether to emit warnings if thresholds exceeded

        Returns:
            dict with resource usage information
        """
        buffer_count = len(self.buffers)
        array_count = len(self.arrays)
        leaked_count = self.count_leaked_buffers()

        usage = {
            "buffer_count": buffer_count,
            "array_count": array_count,
            "leaked_count": leaked_count,
            "purged_total": self.purged,
        }

        if warn:
            if buffer_count > self.BUFFER_WARNING_THRESHOLD:
                warnings.warn(
                    f"High buffer count detected: {buffer_count} buffers allocated "
                    f"(threshold: {self.BUFFER_WARNING_THRESHOLD}). "
                    f"Consider calling purge() to free unused buffers.",
                    ResourceWarning,
                    stacklevel=2,
                )

            if leaked_count > self.LEAKED_BUFFER_WARNING_THRESHOLD:
                warnings.warn(
                    f"Memory leak detected: {leaked_count} buffers without array references "
                    f"(threshold: {self.LEAKED_BUFFER_WARNING_THRESHOLD}). "
                    f"Call purge() to clean up orphaned buffers.",
                    ResourceWarning,
                    stacklevel=2,
                )

        return usage

    def reset(self):
        """Reset statistics counters."""
        self.purged = self.hits = self.misses = 0

    def snapshot(self, **kwargs) -> dict:
        """
        Take a snapshot of current statistics.

        Args:
            **kwargs: Additional metadata to include in snapshot

        Returns:
            dict: Snapshot of current statistics
        """
        snap = self.stats(**kwargs)
        self.snapshots.append(snap)
        return snap

    def purge_buffers(self):
        """
        Purge device buffers that no longer have numpy array references.

        Uses weak references to automatically clean up buffers when
        the corresponding numpy arrays are garbage collected.
        """
        for buffer_id in list(self.buffers.keys()):
            if buffer_id not in self.arrays:
                # Get buffer reference before deleting
                buffer = self.buffers[buffer_id]
                # Give backend a chance to explicitly free resources
                self._free_buffer(buffer)
                # Remove from tracking
                del self.buffers[buffer_id]
                self.purged += 1

    def _free_buffer(self, buffer: Any):
        """
        Explicitly free device buffer resources.

        Override in subclass for backend-specific cleanup.
        Default implementation relies on Python GC.

        Args:
            buffer: Backend-specific buffer object to free
        """
        # Default: just let Python GC handle it
        pass

    def purge(self):
        """Purge unused device buffers (convenience wrapper)."""
        self.purge_buffers()

    def __enter__(self):
        """Context manager entry - return self for use in with statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cleanup on exit."""
        # Purge all buffers when exiting context
        self.purge_buffers()
        return False  # Don't suppress exceptions

    def close(self):
        """
        Explicitly close and cleanup device manager resources.

        This method purges all buffers and allows backend-specific cleanup.
        Safe to call multiple times.
        """
        # Purge all tracked buffers
        self.purge_buffers()
        # Allow backend-specific cleanup
        self._close()

    def _close(self):
        """
        Backend-specific cleanup operations.

        Override in subclass for backend-specific resource cleanup.
        Default implementation does nothing.
        """
        pass

    @abstractmethod
    def make_buffer(self, a: Array) -> Any:
        """
        Create a new device buffer from numpy array.

        Args:
            a: Numpy array to create buffer from

        Returns:
            Backend-specific buffer object

        Raises:
            RuntimeError: If array is not contiguous
        """
        pass

    def ensure_buffer(self, a: Array) -> Any:
        """
        Ensure a device buffer exists for array, creating if necessary.

        Args:
            a: Numpy array

        Returns:
            Backend-specific buffer object
        """
        buf = self.find_buffer(a, self.WRITE)
        if buf is None:
            buf = self.make_buffer(a)
        return buf

    def ensure(self, a: Array) -> Any:
        """
        Ensure device buffer exists for array (convenience wrapper).

        Args:
            a: Numpy array

        Returns:
            Backend-specific buffer object
        """
        return self.ensure_buffer(a)

    @abstractmethod
    def read_buffer(self, a: Array) -> Array:
        """
        Read data from device buffer back to numpy array.

        Args:
            a: Numpy array to read into

        Returns:
            The numpy array with data from device buffer
        """
        pass

    def read(self, a: Array) -> Array:
        """
        Read device buffer back to numpy array (convenience wrapper).

        Args:
            a: Numpy array to read into

        Returns:
            The numpy array with data from device
        """
        return self.read_buffer(a)

    @abstractmethod
    def write_buffer(self, a: Array, wait: bool = True) -> Any:
        """
        Write numpy array data to device buffer.

        Args:
            a: Numpy array to write
            wait: Whether to wait for operation to complete

        Returns:
            Backend-specific buffer object
        """
        pass

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

    @abstractmethod
    def find_buffer(self, a: Array, op: int) -> Any | None:
        """
        Find existing device buffer for array.

        Args:
            a: Numpy array to find buffer for
            op: Operation type (READ or WRITE)

        Returns:
            Backend-specific buffer object, or None if not found

        Raises:
            ValueError: If reading an array with no existing buffer
        """
        pass

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
            _device_managers[device_name] = dmgr = backend_module.DeviceManager(
                device_name
            )
        return dmgr

    return get_device_manager
