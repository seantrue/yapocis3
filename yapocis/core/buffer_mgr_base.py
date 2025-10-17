#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.

"""
Abstract base class for backend-specific buffer managers.

This module defines BufferManagerBase, which establishes the common interface
for managing GPU/device memory buffers across all backends (OpenCL, Metal, CUDA, Numba).

All backend buffer managers must implement:
  - make_buffer(a) - Create a new device buffer from numpy array
  - read_buffer(a) - Read device buffer back to numpy array
  - write_buffer(a, wait) - Write numpy array to device buffer
  - find_buffer(a, op) - Find existing buffer for array

Common functionality provided by base class:
  - Statistics tracking (hits, misses, purged buffers)
  - Performance timing (optional)
  - Buffer lifecycle management
  - Snapshot/stats reporting
"""

from abc import ABC, abstractmethod
from weakref import WeakValueDictionary as Weak
import time
from typing import Any

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


class BufferManagerBase(ABC):
    """
    Abstract base class for backend-specific buffer managers.

    Manages the lifecycle of device memory buffers, tracking usage statistics
    and providing common buffer management operations.

    Class constants:
        READ: Operation type for reading from device
        WRITE: Operation type for writing to device
        DEBUG: Enable debug output
        TIMED: Enable performance timing
        TIMED_METHODS: Methods to time when TIMED=True

    Subclasses must implement:
        - make_buffer(a) -> backend-specific buffer type
        - read_buffer(a) -> Array
        - write_buffer(a, wait=True) -> backend-specific buffer type
        - find_buffer(a, op) -> backend-specific buffer type | None
    """

    READ = 0
    WRITE = 1
    DEBUG = True
    TIMED = False
    TIMED_METHODS = ("make_buffer", "ensure_buffer", "read_buffer", "write_buffer")

    def __init__(self, context: Any, queue_or_stream: Any, timed: bool = False, autoflatten: bool = True):
        """
        Initialize buffer manager.

        Args:
            context: Backend-specific context (pyopencl.Context, pycuda.Context, etc)
            queue_or_stream: Backend-specific queue/stream for async operations
            timed: Enable performance timing
            autoflatten: Automatically flatten arrays for buffer operations
        """
        self.ctx = context
        self.queue = queue_or_stream  # Generic name for queue/stream
        self.autoflatten = autoflatten

        # Buffer tracking
        self.arrays = Weak()  # Weak references to numpy arrays
        self.buffers = {}     # Strong references to device buffers

        # Statistics
        self.hits = self.misses = 0
        self.purged = 0
        self.snapshots = []

        # Performance timing
        self.timed = timed
        self.times = {name: 0 for name in self.TIMED_METHODS}
        if self.timed or self.TIMED:
            for name in self.TIMED_METHODS:
                setattr(self, name, method_timer(self, name, getattr(self, name)))

    def stats(self, **kwargs) -> dict:
        """
        Get current buffer manager statistics.

        Returns:
            dict: Statistics including hits, misses, purged count, buffer count, and timings
        """
        d = dict(
            purged=self.purged,
            hits=self.hits,
            misses=self.misses,
            buffers=len(self.buffers)
        )
        d.update(self.times)
        d.update(kwargs)
        return d

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
                del self.buffers[buffer_id]
                self.purged += 1

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
