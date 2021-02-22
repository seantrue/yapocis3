#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import time
from typing import Callable, List, Tuple, Any, Union

import numpy as np
import pyopencl

from yapocis.utils.typing import Array


class Kernel:
    """
    Provide a callable interface to an OpenCL kernel function.
    """

    def __init__(self, kernel_manager: "KernelManager", kernel: Callable, params: List[Tuple]):
        self.kernel_manager = kernel_manager
        self.queue = kernel_manager.dmgr.queue
        self.kernel = kernel
        self.params = params
        self.autoflatten = kernel_manager.dmgr.autoflatten
        self.argspecs = self.posbyname = self.returns = self.global_size = None
        self.map_arguments()

    def read(self, a: Array):
        return self.kernel_manager.dmgr.read(a)

    def write(self, a: Array):
        return self.kernel_manager.dmgr.write(a)

    def ensure(self, a: Array):
        return self.kernel_manager.dmgr.ensure(a)

    def map_arguments(self):
        argspecs = [None] * len(self.params)
        posbyname = {}
        iarg = 0
        for iparam, (bufferHint, dtype, isbuffer, name) in enumerate(self.params):
            if bufferHint in ("outlike", "sizeof", "widthof", "heightof", "depthof"):
                argspecs[iparam] = (bufferHint, name)
            else:
                posbyname[name] = iarg
                argspecs[iparam] = (bufferHint, iarg)
                iarg += 1
        self.argspecs = argspecs
        self.posbyname = posbyname

    def prepare_arguments(self, args:List[Any]):
        "Takes the actual arguments and maps to the arguments needed by the kernel"
        assert len(self.posbyname) == len(args), f" Got {len(args)}, expected {len(self.posbyname)} arguments, [{self.posbyname}]"
        self.returns = []
        realargs = [None] * len(self.params)
        for iparam, (bufferHint, value) in enumerate(self.argspecs):
            (_, data_type, isbuffer, name) = self.params[iparam]
            if data_type in ("char", "uchar"):
                dtype = "uint8"
            if data_type:
                dtype = getattr(np, data_type)
            else:
                dtype = None
            if not isbuffer:
                assert dtype, "Must have type for scalar argument %s" % iparam
                arg = args[value]
                if dtype in (np.float32, np.float64):
                    dtype = np.float32
                real_arg = np.asscalar(dtype(arg))
                if isinstance(real_arg,tuple):
                    real_arg = real_arg[0]
                realargs[iparam] = real_arg
                continue
            methodname = "handle_%s" % bufferHint
            realargs[iparam] = getattr(self, methodname)(args, value, dtype)
        if self.global_size is None:
            for arg in args:
                if hasattr(arg, "size"):
                    if self.autoflatten:
                        self.global_size = arg.size
                    else:
                        self.global_size = arg.shape
                    break
        return tuple(realargs)

    def handle_sizeof(self, args, name, dtype):
        arg = args[self.posbyname[name]]
        if not dtype:
            dtype = np.int32
        return dtype(arg.size)

    # Assume row major storage.
    # addr(x,y) = (x*height)+y
    #  x = addr/height
    #  y = (addr - (addr/height)*height)
    def handle_widthof(self, args, name, dtype):
        arg = args[self.posbyname[name]]
        if not dtype:
            dtype = np.int32
        return dtype(arg.shape[0])

    def handle_heightof(self, args, name, dtype):
        arg = args[self.posbyname[name]]
        if not dtype:
            dtype = np.int32
        return dtype(arg.shape[1])

    def handle_depthof(self, args, name, dtype):
        arg = args[self.posbyname[name]]
        if not dtype:
            dtype = np.int32
        return dtype(arg.shape[2])

    def handle_outlike(self, args, name, dtype):
        arg = args[self.posbyname[name]]
        if not dtype:
            dtype = arg.dtype
        a = np.zeros_like(arg, dtype=dtype)
        buf = self.ensure(a)
        self.returns.append((a, buf))
        return buf

    def handle_in(self, args, pos, dtype):
        arg = args[pos]
        assert dtype
        if arg.dtype != dtype:
            arg = arg.astype(dtype)
        buf = self.write(arg)
        return buf

    def handle_inout(self, args, pos, dtype):
        arg = args[pos]
        assert dtype
        if arg.dtype != dtype:
            arg = arg.astype(dtype)
        buf = self.write(arg)
        self.returns.append((arg, buf))
        return buf

    def handle_out(self, args, pos, dtype):
        arg = args[pos]
        assert dtype
        if arg.dtype != dtype:
            arg = arg.astype(dtype)
        buf = self.ensure(arg)
        self.returns.append((arg, buf))
        return buf

    handle_resident = handle_out

    def __call__(self, *args: Tuple[Any], **kwargs):
        """Call a kernel, prepare arguments, and track performance"""
        self.kernel_manager.calls += 1
        t = time.time()
        self.global_size = kwargs.pop("global_size", None)
        self.local_size = kwargs.pop("local_size", None)
        prepared_args = self.prepare_arguments(args)
        def show_args():
            for iarg, arg in enumerate(args):
                print(f"Argument: {iarg+1}: {type(arg)} {str(arg)[:64]}")
            for iarg, arg in enumerate(prepared_args):
                print(f"Argument: {iarg+1}: {type(arg)} {str(arg)[:64]}")
        if not isinstance(self.global_size, (tuple, list)):
            self.global_size = (self.global_size,)
        try:
            self.evt = self.kernel(self.kernel_manager.dmgr.queue, self.global_size, self.local_size, *prepared_args)
        except pyopencl.LogicError:
            print(f"Run time error calling {self.name}")
            show_args()
            raise RuntimeError(f"Logic error calling {self.name}")
        except RuntimeError:
            print(f"Logic error calling {self.name}")
            show_args()
            raise
        poll = kwargs.pop("poll", None)
        if poll:
            while True:
                status = self.evt.get_info(pyopencl.event_info.COMMAND_EXECUTION_STATUS)
                if status == pyopencl.command_execution_status.COMPLETE:
                    break
                time.sleep(poll)
        else:
            self.evt.wait()
        # pyopencl.enqueue_barrier(self.queue)
        rval = self.prepare_return()
        self.kernel_manager.runtime += (time.time() - t)
        return rval

    def prepare_return(self) -> Union[Any, Tuple[Any], None]:
        """Takes the return value buffers (out and outlike) and prepares proper Numpy arrays"""
        rvals = []
        for arg, buf in self.returns:
            self.read(arg)
            rvals.append(arg)
        self.returns = []
        if len(rvals) == 0:
            return None
        elif len(rvals) == 1:
            return rvals[0]
        return tuple(rvals)
