#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.

"""
Abstract base class for backend-specific device managers.

This module defines DeviceManagerBase, which establishes the common interface
for managing compute devices and their buffers across all backends (OpenCL, Metal, CUDA, Numba).

All backend device managers must implement:
  - build(source, options) - Compile kernel source for device
  - __init__(device_name, timed, **kwargs) - Initialize device and context
  - make_buffer(a) - Create a new device buffer from numpy array, and record it
    with ``_register(a, buf)``
  - read_buffer(a) - Read device buffer back to numpy array
  - write_buffer(a, wait) - Write numpy array to device buffer

``find_buffer(a, op)`` is concrete in the base class (#100); backends don't
override it.

Common functionality provided by base class:
  - Buffer management (write, read, ensure, find, purge)
  - Statistics tracking (hits, misses, purged and released buffers)
  - Performance timing (optional)
  - Snapshot/stats reporting

The DeviceManager combines:
  - Device selection and initialization
  - Context/queue/stream management
  - Buffer management with weak reference tracking
  - Kernel compilation

Buffer lifecycle (#100):
  - One store, ``_store``, maps ``id(a)`` to ``(weakref.KeyedRef(a), buffer)``.
    Backends add entries only through ``_register(a, buf)``; lookups go through
    ``_lookup(a)``, which ``find_buffer`` uses.
  - A buffer is found for ``a`` only when the entry under ``id(a)`` holds a weak
    reference whose referent is ``a``. ``id()`` is reused after an array dies, so
    the key alone never identifies an array.
  - When an array dies, its ref's callback appends the ref to the ``_dead`` deque
    (lock-free, no device calls). The next manager call (register, lookup, find,
    read, write, ensure, purge, stats) drains the deque and drops the dead
    arrays' buffers ("fast free"), counting them in ``released``. ``purge()`` is
    no longer needed to free memory; it remains as a safety net.
  - ``arrays`` and ``buffers`` are read-only views of the store.
  - ``_lock`` (an RLock) is held wherever the store is mutated or iterated.
"""

import collections
import contextlib
import threading
import time
import warnings
import weakref
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
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


class _BuffersView(Mapping):
    """Read-only ``id -> buffer`` view of a device manager's store.

    Includes entries whose array has died but that haven't been drained yet.
    """

    __slots__ = ("_mgr",)

    def __init__(self, mgr: "DeviceManagerBase"):
        self._mgr = mgr

    def _entries(self) -> list[tuple[int, tuple[weakref.KeyedRef, Any]]]:
        with self._mgr._lock:
            return list(self._mgr._store.items())

    def __getitem__(self, key: int) -> Any:
        return self._mgr._store[key][1]

    def __contains__(self, key: object) -> bool:
        return key in self._mgr._store

    def __iter__(self) -> Iterator[int]:
        return iter([k for k, _ in self._entries()])

    def __len__(self) -> int:
        return len(self._mgr._store)

    def keys(self) -> list[int]:  # type: ignore[override]
        return [k for k, _ in self._entries()]

    def values(self) -> list[Any]:  # type: ignore[override]
        return [e[1] for _, e in self._entries()]

    def items(self) -> list[tuple[int, Any]]:  # type: ignore[override]
        return [(k, e[1]) for k, e in self._entries()]

    def __repr__(self) -> str:
        return f"<buffers view: {len(self)} entries>"


class _ArraysView(Mapping):
    """Read-only ``id -> array`` view of a device manager's store.

    Only arrays that are still alive are visible, so an entry disappears as soon
    as its array dies, with no manager call (as with a WeakValueDictionary).
    """

    __slots__ = ("_mgr",)

    def __init__(self, mgr: "DeviceManagerBase"):
        self._mgr = mgr

    def _live(self) -> list[tuple[int, Array]]:
        with self._mgr._lock:
            entries = list(self._mgr._store.items())
        live = []
        for k, (ref, _) in entries:
            a = ref()
            if a is not None:
                live.append((k, a))
        return live

    def __getitem__(self, key: int) -> Array:
        a: Array | None = self._mgr._store[key][0]()
        if a is None:
            raise KeyError(key)
        return a

    def __contains__(self, key: object) -> bool:
        e = self._mgr._store.get(key)  # type: ignore[call-overload]
        return e is not None and e[0]() is not None

    def __iter__(self) -> Iterator[int]:
        return iter([k for k, _ in self._live()])

    def __len__(self) -> int:
        return len(self._live())

    def keys(self) -> list[int]:  # type: ignore[override]
        return [k for k, _ in self._live()]

    def values(self) -> list[Array]:  # type: ignore[override]
        return [a for _, a in self._live()]

    def items(self) -> list[tuple[int, Array]]:  # type: ignore[override]
        return self._live()

    def __repr__(self) -> str:
        return f"<arrays view: {len(self)} live arrays>"


class DeviceManagerBase(ABC):
    """
    Abstract base class for backend-specific device managers.

    Manages both device operations and buffer lifecycle, tracking usage statistics
    and providing common buffer management operations.

    Buffer identity: a buffer is found for ``a`` only when the store entry under
    ``id(a)`` holds a weak reference whose referent is ``a``. ``id()`` is reused
    after an array dies, so the key alone never identifies an array. Never index
    ``_store`` (or the ``buffers`` view) by ``id(x)`` to find a buffer; go through
    ``_lookup``. Backends register buffers only through ``_register``.

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

    Attributes:
        device_name: String identifier for the device
        device: Backend-specific device object
        ctx: Backend-specific context object
        queue/stream: Backend-specific queue or stream for async operations
        context_meta: Dictionary for storing additional context metadata
        arrays: Read-only view, ``id -> array`` for live arrays in the store
        buffers: Read-only view, ``id -> buffer`` for every entry in the store
        hits/misses: Buffer cache statistics
        purged: Count of buffers dropped by an explicit purge
        released: Count of buffers dropped by fast free (array died, then drained)
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

        # Buffer tracking: id(a) -> (weak ref to a, device buffer). The only table.
        self._store: dict[int, tuple[weakref.KeyedRef, Any]] = {}
        # Refs whose arrays died, appended by the weakref callback, drained on the
        # next manager call.
        self._dead: collections.deque = collections.deque()
        # Held wherever the store is mutated or iterated (re-entrant: the drain is
        # called from methods that already hold it).
        self._lock = threading.RLock()

        # Statistics
        self.hits = self.misses = 0
        self.purged = 0
        self.released = 0
        self.snapshots = []

        # Performance timing
        self.timed = timed
        self.times = dict.fromkeys(self.TIMED_METHODS, 0)
        if self.timed or self.TIMED:
            for name in self.TIMED_METHODS:
                setattr(self, name, method_timer(self, name, getattr(self, name)))

    # --- Store: views, registration, lookup, drain -------------------------------

    @property
    def arrays(self) -> Mapping[int, Array]:
        """Read-only view: ``id -> array`` for the live arrays in the store."""
        return _ArraysView(self)

    @property
    def buffers(self) -> Mapping[int, Any]:
        """Read-only view: ``id -> buffer`` for every entry in the store."""
        return _BuffersView(self)

    def _register(self, a: Array, buf: Any) -> None:
        """Record ``buf`` as the device buffer of ``a``. The only way to add an entry.

        Reuses the entry's weak ref when it already refers to ``a`` (Metal makes a
        new buffer on every write), so a live array has exactly one ref.
        """
        with self._lock:
            if self._dead:
                self._drain()
            k = id(a)
            e = self._store.get(k)
            if e is not None and e[0]() is a:
                ref = e[0]
            else:
                ref = weakref.KeyedRef(a, self._dead.append, k)
            self._store[k] = (ref, buf)

    def _lookup(self, a: Array) -> Any | None:
        """Return the buffer registered for ``a``, or None.

        A buffer is found for ``a`` only when the store entry under ``id(a)`` holds
        a weak reference whose referent is ``a``. ``id()`` is reused after an array
        dies, so the key alone never identifies an array. Never index ``_store``
        (or the ``buffers`` view) by ``id(x)`` to find a buffer; go through
        ``_lookup``.

        Takes no lock: the read is one ``dict.get`` of an immutable tuple, and the
        guard rejects any entry that doesn't belong to ``a``.
        """
        if self._dead:
            self._drain()
        e = self._store.get(id(a))
        if e is not None and e[0]() is a:
            return e[1]
        return None

    def _drain(self) -> None:
        """Drop the buffers of arrays that died since the last drain (fast free).

        Each queued ref is matched against the store entry under its key; an entry
        that is missing (cleared or purged) or holds a different ref (a new array
        took the id) is left alone. The store is updated before anything is
        released, so a re-entrant call during the release sees a consistent store.
        """
        release: list[Any] = []
        with self._lock:
            dead = self._dead
            store = self._store
            while dead:
                ref = dead.popleft()
                k = ref.key
                e = store.get(k)
                if e is not None and e[0] is ref:
                    del store[k]
                    release.append(e[1])
                e = None
            self.released += len(release)
        if release:
            with self._release_context():
                for buf in release:
                    self._free_buffer(buf)
                buf = None
                release.clear()

    def _release_context(self) -> contextlib.AbstractContextManager[Any]:
        """Context in which buffers are released. CUDA makes its context current."""
        return contextlib.nullcontext()

    def _clear_store(self) -> None:
        """Forget every entry and queued ref, dropping the buffers in the caller's context."""
        with self._lock:
            entries = list(self._store.values())
            self._store.clear()
            self._dead.clear()
        entries.clear()

    def _counts(self) -> tuple[int, int, int]:
        """Drain, then return (buffers, live arrays, dead entries) from the store."""
        self._drain()
        with self._lock:
            refs = [ref for ref, _ in self._store.values()]
        live = sum(1 for ref in refs if ref() is not None)
        return len(refs), live, len(refs) - live

    # --- Statistics ---------------------------------------------------------------

    def stats(self, **kwargs) -> dict:
        """
        Get current buffer manager statistics.

        Drains dead arrays' entries first, so counts reflect live arrays.

        Returns:
            dict: Statistics including hits, misses, purged and released counts,
            buffer, array and leaked counts, and timings
        """
        buffers, arrays, leaked = self._counts()
        d = dict(
            purged=self.purged,
            released=self.released,
            hits=self.hits,
            misses=self.misses,
            buffers=buffers,
            arrays=arrays,
            leaked=leaked,
        )
        d.update(self.times)
        d.update(kwargs)
        return d

    def count_leaked_buffers(self) -> int:
        """
        Count buffers that no longer have corresponding array references.

        Drains first, so this is 0 after any manager call unless a ref's callback
        was lost.

        Returns:
            Number of leaked buffers (buffers without array references)
        """
        return self._counts()[2]

    def check_resource_usage(self, warn: bool = True) -> dict:
        """
        Check current resource usage and optionally warn if thresholds exceeded.

        Args:
            warn: Whether to emit warnings if thresholds exceeded

        Returns:
            dict with resource usage information
        """
        buffer_count, array_count, leaked_count = self._counts()

        usage = {
            "buffer_count": buffer_count,
            "array_count": array_count,
            "leaked_count": leaked_count,
            "purged_total": self.purged,
            "released_total": self.released,
        }

        if warn:
            if buffer_count > self.BUFFER_WARNING_THRESHOLD:
                warnings.warn(
                    f"High buffer count detected: {buffer_count} buffers allocated "
                    f"(threshold: {self.BUFFER_WARNING_THRESHOLD}). "
                    f"Each belongs to a live array; buffers are released when "
                    f"their arrays die.",
                    ResourceWarning,
                    stacklevel=2,
                )

            if leaked_count > self.LEAKED_BUFFER_WARNING_THRESHOLD:
                warnings.warn(
                    f"Memory leak detected: {leaked_count} buffers without array references "
                    f"(threshold: {self.LEAKED_BUFFER_WARNING_THRESHOLD}). "
                    f"purge() drops them.",
                    ResourceWarning,
                    stacklevel=2,
                )

        return usage

    def reset(self):
        """Reset statistics counters."""
        self.purged = self.hits = self.misses = 0
        self.released = 0

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

        Buffers are normally released at the next manager call after their array
        dies (see ``_drain``), so this is a safety net: it drains, then drops any
        entry whose array is dead, counting those in ``purged``. The store is
        updated before anything is released.
        """
        release: list[Any] = []
        with self._lock:
            if self._dead:
                self._drain()
            for k, e in list(self._store.items()):
                if e[0]() is None:
                    del self._store[k]
                    release.append(e[1])
            e = None
            self.purged += len(release)
            if release:
                with self._release_context():
                    for buf in release:
                        # Give backend a chance to explicitly free resources
                        self._free_buffer(buf)
                    buf = None
                    release.clear()

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

    def make_output_buffer(self, a: Array) -> Any:
        """
        Create a device buffer for a write-only array, without uploading it.

        The default uploads like ``make_buffer``; backends override it with an
        uninitialised allocation. Only for ``out`` and ``out_like`` arrays,
        whose kernels write every element and never read it. A kernel that
        accumulates into its output must declare it ``inout``.
        """
        return self.make_buffer(a)

    def ensure_output_buffer(self, a: Array) -> Any:
        """
        Like ``ensure_buffer``, but a new buffer is made with
        ``make_output_buffer``. An existing buffer is reused as is.
        """
        buf = self.find_buffer(a, self.WRITE)
        if buf is None:
            buf = self.make_output_buffer(a)
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

    def find_buffer(self, a: Array, op: int) -> Any | None:
        """
        Find existing device buffer for array, through the guarded ``_lookup``.

        Args:
            a: Numpy array to find buffer for
            op: Operation type (READ or WRITE)

        Returns:
            Backend-specific buffer object, or None if not found

        Raises:
            ValueError: If reading an array with no existing buffer
        """
        assert op in (
            self.READ,
            self.WRITE,
        ), f"{op=} not one of {self.READ=},{self.WRITE=}"
        buf = self._lookup(a)
        if buf is not None:
            self.hits += 1
            return buf
        self.misses += 1
        if op == self.READ:
            buffer_id = id(a)
            raise ValueError(
                f"Array {buffer_id=} not in yapocis management, you may have written to it, or be using an assigned or .copy"
            )
        return None

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
