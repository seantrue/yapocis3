from __future__ import annotations

"""
Buffer management for PyMetallic backend.

This mirrors the OpenCL `BufferManager` but calls into `pymetallic` for:
  - Context, CommandQueue
  - Buffer
  - enqueue_copy
"""

import time
from weakref import WeakValueDictionary as Weak

try:
    import pymetallic  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyMetallic is required for the metallic backend") from e

from yapocis.utils.stereo_typing import *  # Array, etc.


def method_timer(self, name, method):
    def timed_method(*args, **kwargs):
        t0 = time.time()
        value = method(*args, **kwargs)
        self.times[name] += time.time() - t0
        return value

    return timed_method


class BufferManager:
    # Default flags if PyMetallic exposes OpenCL-like mem_flags (not typical)
    MEMFLAGS = getattr(pymetallic, "mem_flags", None)
    if MEMFLAGS is not None:
        MEMFLAGS = pymetallic.mem_flags.READ_WRITE | pymetallic.mem_flags.COPY_HOST_PTR
    READ = 0
    WRITE = 1
    DEBUG = True
    TIMED = False
    TIMED_METHODS = ("make_buffer", "ensure_buffer", "read_buffer", "write_buffer")

    def __init__(
        self,
        device: "pymetallic.Device",
        queue: "pymetallic.CommandQueue" | None,
        timed: bool = False,
    ):
        self.device = device
        self.queue = device.make_command_queue() if queue is None else queue
        # Buffer management
        self.arrays = Weak()
        self.buffers: dict[int, "pymetallic.Buffer"] = {}
        self.hits = self.misses = 0
        self.purged = 0
        self.snapshots: list[dict] = []
        self.timed = timed
        self.times = {name: 0.0 for name in self.TIMED_METHODS}
        if self.timed or self.TIMED:
            for name in self.TIMED_METHODS:
                setattr(self, name, method_timer(self, name, getattr(self, name)))

    def stats(self, **kwargs):
        d = dict(
            purged=self.purged,
            hits=self.hits,
            misses=self.misses,
            buffers=len(self.buffers),
        )
        d.update(self.times)
        d.update(kwargs)
        return d

    def reset(self):
        self.purged = self.hits = self.misses = 0

    def snapshot(self, **kwargs):
        snap = self.stats(**kwargs)
        self.snapshots.append(snap)
        return snap

    def purge_buffers(self):
        for buffer_id in list(self.buffers.keys()):
            if buffer_id not in self.arrays:
                del self.buffers[buffer_id]
                self.purged += 1

    def make_buffer(self, a: Array) -> "pymetallic.Buffer":
        # Ensure contiguous data for low-level transfer
        if not a.flags.c_contiguous:
            a = np.ascontiguousarray(a)
        buf = pymetallic.Buffer.from_numpy(self.device, a)
        buffer_id = id(a)
        self.arrays[buffer_id] = a
        self.buffers[buffer_id] = buf
        return buf

    def ensure_buffer(self, a: Array) -> "pymetallic.Buffer":
        buf = self.find_buffer(a, self.WRITE)
        if buf is None:
            buf = self.make_buffer(a)
        return buf

    def read_buffer(self, a: Array) -> Array:
        buf = self.find_buffer(a, self.READ)
        view = buf.to_numpy(dtype=a.dtype, shape=a.shape)
        np.copyto(a, view)
        return a

    def write_buffer(self, a: Array, wait: bool = True) -> "pymetallic.Buffer":
        # For PyMetallic we create/replace the backing buffer from host bytes
        if not a.flags.c_contiguous:
            a = np.ascontiguousarray(a)
        buf = pymetallic.Buffer.from_numpy(self.device, a)
        buffer_id = id(a)
        self.arrays[buffer_id] = a
        self.buffers[buffer_id] = buf
        return buf

    def find_buffer(self, a: Array, op: int) -> "pymetallic.Buffer | None":
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
