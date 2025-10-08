#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.

import time
from weakref import WeakValueDictionary as Weak

import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

from yapocis.utils.stereo_typing import *


def method_timer(self, name, method):
    def timed_method(*args, **kwargs):
        t = time.time()
        value = method(*args, **kwargs)
        t = time.time() - t
        self.times[name] += t
        return value

    return timed_method


class BufferManager:
    READ = 0
    WRITE = 1
    DEBUG = True
    TIMED = False
    TIMED_METHODS = ("make_buffer", "ensure_buffer", "read_buffer", "write_buffer")

    def __init__(self, context: cuda.Context, stream: cuda.Stream, timed: bool = False, autoflatten: bool = True):
        self.ctx = context
        self.stream = stream
        # Buffer management
        self.autoflatten = autoflatten
        self.arrays = Weak()
        self.buffers = {}
        self.hits = self.misses = 0
        self.purged = 0
        self.snapshots = []
        self.timed = timed
        self.times = {name: 0 for name in self.TIMED_METHODS}
        if self.timed or self.TIMED:
            for name in self.TIMED_METHODS:
                setattr(self, name, method_timer(self, name, getattr(self, name)))

    def stats(self, **kwargs):
        d = dict(purged=self.purged, hits=self.hits, misses=self.misses, buffers=len(self.buffers))
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
        # Drop buffers if no array or data buffer is referencing them
        for buffer_id in list(self.buffers.keys()):
            if buffer_id not in self.arrays:
                del self.buffers[buffer_id]
                self.purged += 1

    def make_buffer(self, a: Array) -> gpuarray.GPUArray:
        if not a.data.contiguous:
            raise RuntimeError(f"Array {id(a)} {a.shape} is not contiguous")
        # Create a GPUArray from the numpy array
        buf = gpuarray.to_gpu_async(a, stream=self.stream)
        buffer_id = id(a)
        self.arrays[buffer_id] = a
        self.buffers[buffer_id] = buf
        return buf

    def ensure_buffer(self, a: Array) -> gpuarray.GPUArray:
        buf = self.find_buffer(a, self.WRITE)
        if buf is None:
            buf = self.make_buffer(a)
        return buf

    def read_buffer(self, a: Array) -> Array:
        buf = self.find_buffer(a, self.READ)
        # Copy from GPU to host
        buf.get_async(self.stream, ary=a)
        self.stream.synchronize()
        return a

    def write_buffer(self, a: Array, wait=True) -> gpuarray.GPUArray:
        buf = self.ensure_buffer(a)
        # Copy from host to GPU
        buf.set_async(a, stream=self.stream)
        if wait:
            self.stream.synchronize()
        return buf

    def find_buffer(self, a: Array, op: int) -> gpuarray.GPUArray | None:
        "Find an appropriate buffer. Tricky."
        assert op in (self.READ, self.WRITE)
        buffer_id = id(a)
        havea = buffer_id in self.buffers and buffer_id in self.arrays
        # Complete match, easy decision
        if havea:
            self.hits += 1
            return self.buffers[buffer_id]
        else:
            self.misses += 1
        # No match at all, also easy.
        if not havea:
            # Reading an array back with no matching buffer is fatal
            if op == self.READ:
                raise ValueError(
                    "Array not in yapocis management, you may have written to it, or be using an assigned or .copy"
                )
            return None
        raise RuntimeError(f"Failed to find a buffer for {op} {a}")
