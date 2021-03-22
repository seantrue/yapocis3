#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from weakref import WeakValueDictionary as Weak
import time
import pyopencl
from pyopencl import mem_flags as mf

from yapocis.utils.stereo_typing import *


# Numpy makes a big deal out of .copy() being a deepcopy. Hah!
# At the time of this writing (2011)
# a.data was shared until a copy is forced by copy on write.
# This makes mapping arrays to remote buffers tricky
# Theona has an interesting approach, which I am ignoring
# in favor of better sugar.

def method_timer(self, name, method):
    def timed_method(*args, **kwargs):
        t = time.time()
        value = method(*args, **kwargs)
        t = time.time()-t
        self.times[name] += t
        return value
    return timed_method

class BufferManager:
    MEMFLAGS = mf.READ_WRITE | mf.COPY_HOST_PTR
    READ = 0
    WRITE = 1
    DEBUG = True
    TIMED = False
    TIMED_METHODS = ("make_buffer","ensure_buffer","read_buffer","write_buffer")
    def __init__(self, context: pyopencl.Context, queue: pyopencl.CommandQueue, timed:bool=False):
        self.ctx = context
        self.queue = pyopencl.CommandQueue(self.ctx) if queue is None else queue
        # Buffer management
        self.arrays = Weak()
        self.buffers = {}
        self.hits = self.misses = 0
        self.purged = 0
        self.snapshots = []
        self.timed = timed
        self.times = {name:0 for name in self.TIMED_METHODS}
        if self.timed or self.TIMED:
            for name in self.TIMED_METHODS:
                setattr(self, name, method_timer(self, name, getattr(self, name)))

    def stats(self, **kwargs):
        d = dict(purged=self.purged, hits=self.hits, misses=self.misses,
                    buffers=len(self.buffers))
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
            if not buffer_id in self.arrays:
                del self.buffers[buffer_id]
                self.purged += 1

    def make_buffer(self, a:Array) -> pyopencl.Buffer:
        if not a.data.contiguous:
            raise RuntimeError(f"Array {id(a)} {a.shape} is not contiguous")
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

    def write_buffer(self, a:Array, wait=True) -> pyopencl.Buffer:
        buf = self.ensure_buffer(a)
        shape = a.shape
        strides = a.strides
        a.shape = (a.size,)
        a.strides = (strides[-1],)
        if wait:
            pyopencl.enqueue_copy(self.queue, buf, a).wait()
        else:
            pyopencl.enqueue_copy(self.queue, buf, a)
        a.shape = shape
        a.strides = strides
        # pyopencl.enqueue_barrier(self.queue)
        return buf

    def find_buffer(self, a:Array, op:int) -> Optional[pyopencl.Buffer]:
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
                    "Array not in yapocis management, you may have written to it, or be using an assigned or .copy")
            return None
        raise RuntimeError(f"Failed to find a buffer for {op} {a}")
