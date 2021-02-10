#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import os
from weakref import WeakValueDictionary as Weak

import pyopencl
from pyopencl import mem_flags as mf

from yapocis.lib.yapocis_types import *


# Numpy makes a big deal out of .copy() being a deepcopy. Hah!
# At the time of this writing (2011)
# a.data was shared until a copy is forced by copy on write.
# This makes mapping arrays to remote buffers tricky
# Theona has an interesting approach, which I am ignoring
# in favor of better sugar.


class BufferManager:
    MEMFLAGS = mf.READ_WRITE | mf.COPY_HOST_PTR
    READ = 0
    WRITE = 1
    DEBUG = True

    def __init__(self, context: pyopencl.Context, queue: Optional[pyopencl.CommandQueue] = None):
        self.ctx = context
        self.queue = queue
        if self.queue is None:
            self.queue = pyopencl.CommandQueue(self.ctx)
        # Buffer management
        self.arrays = Weak()
        self.buffers = {}
        self.hits = self.misses = 0
        self.purged = 0

    def purge_buffers(self):
        # Drop buffers if no array or data buffer is referencing them
        buffer_ids = set(self.arrays.keys())
        for buffer_id in list(self.buffers.keys()):
            if not buffer_id in buffer_ids:
                del self.buffers[buffer_id]
                self.purged += 1

    def make_buffer(self, a:Array) -> pyopencl.Buffer:
        buf = pyopencl.Buffer(self.ctx, self.MEMFLAGS, hostbuf=a)
        buffer_id = id(a)
        self.arrays[buffer_id] = a
        self.buffers[buffer_id] = buf
        return buf

    def ensure_buffer(self, a:Array) -> pyopencl.Buffer:
        buf = self.find_buffer(a, self.WRITE)
        if buf is None:
            buf = self.make_buffer(a)
        return buf

    def read_buffer(self, a:Array) -> pyopencl.Buffer:
        buf = self.find_buffer(a, self.READ)
        shape = a.shape
        strides = a.strides
        a.shape = (a.size,)
        a.strides = (strides[-1],)
        # pyopencl.enqueue_barrier(self.queue)
        pyopencl.enqueue_copy(self.queue, a, buf).wait()
        a.shape = shape
        a.strides = strides
        return buf

    def write_buffer(self, a:Array) -> pyopencl.Buffer:
        buf = self.ensure_buffer(a)
        shape = a.shape
        strides = a.strides
        a.shape = (a.size,)
        a.strides = (strides[-1],)
        pyopencl.enqueue_copy(self.queue, buf, a).wait()
        a.shape = shape
        a.strides = strides
        # pyopencl.enqueue_barrier(self.queue)
        return buf

    def find_buffer(self, a:Array, op:int) -> Optional[pyopencl.Buffer]:
        "Find an appropriate buffer. Tricky."
        assert op in (self.READ, self.WRITE)
        self.purge_buffers()
        buffer_id = id(a)
        havea = buffer_id in self.buffers
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
                    "Array not in yapocis management, you may have written to it, or be using an assigned or .copy")
            return None
        raise RuntimeError(f"Failed to find a buffer for {op} {a}")
