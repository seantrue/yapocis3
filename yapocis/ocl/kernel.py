#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

import time
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pyopencl

from yapocis.utils.stereo_typing import Array


class Kernel:
    """
    Provide a callable interface to an OpenCL kernel function.
    """

    def __init__(self, name, kernel_manager: "KernelManager", kernel: Callable, params: List[Tuple], async_write:bool=False):
        self.name = name
        self.kernel_manager = kernel_manager
        self.queue = kernel_manager.dmgr.queue
        self.kernel = kernel
        self.params = params
        self.autoflatten = kernel_manager.dmgr.autoflatten
        self.argspecs = self.posbyname = self.returns = self.local_size = self.global_size = None
        self.async_write = async_write
        self.map_arguments()

    def read(self, a: Array):
        return self.kernel_manager.dmgr.read(a)

    def write(self, a: Array,wait=True):
        return self.kernel_manager.dmgr.write(a,wait=wait)

    def ensure(self, a: Array):
        return self.kernel_manager.dmgr.ensure(a)

    def map_arguments(self):
        argspecs = [None] * len(self.params)
        posbyname = {}
        iarg = 0
        for iparam, param in enumerate(self.params):
            argspecs[iparam] = param
            if not param.derived:
                posbyname[param.name] = iarg
                iarg += 1
        self.argspecs = argspecs
        self.posbyname = posbyname

    def prepare_arguments(self, args:List[Any], resident_arrays:Optional[List[str]]=None):
        "Takes the actual arguments and maps to the arguments needed by the kernel"
        assert len(self.posbyname) == len(args), f" Got {len(args)}, expected {len(self.posbyname)} arguments, [{self.posbyname}]"
        self.returns = []
        if resident_arrays is not None:
            resident_arrays = [id(ra) for ra in resident_arrays]
        realargs = [None] * len(self.params)
        for iparam, param in enumerate(self.argspecs):
            param = self.params[iparam]
            if not param.np_dtype:
                print(param)
            dtype = getattr(np, param.np_dtype)
            iarg = self.posbyname[param.derived_from]
            if not param.hint and not param.dim:
                arg = args[iarg]
                real_arg = dtype(arg)
            elif param.dim:
                method_name = f"handle_{param.dim}"
                arg = args[iarg]
                real_arg = getattr(self, method_name)(arg, dtype)
            elif param.hint:
                method_name = f"handle_{param.hint}"
                arg = args[iarg]
                if not isinstance(arg, np.ndarray):
                    raise RuntimeError(f"Parameter {param.name} is defined as an array, but is a {type(arg)}")
                if param.hint != "out_like" and resident_arrays and id(arg) in resident_arrays:
                    method_name="handle_resident"
                real_arg = getattr(self, method_name)(arg, dtype)
            realargs[iparam] = real_arg
        if self.global_size is None:
            for arg in args:
                if hasattr(arg, "size"):
                    if self.autoflatten:
                        self.global_size = arg.size
                    else:
                        self.global_size = arg.shape
                    break
        return tuple(realargs)

    def handle_sizeof(self, arg, dtype):
        if not dtype:
            dtype = np.int32
        return dtype(arg.size)
    handle_size_of = handle_sizeof

    def handle_widthof(self, arg, dtype):
        if not dtype:
            dtype = np.int32
        return dtype(arg.shape[0])
    handle_width_of = handle_widthof

    def handle_heightof(self, arg, dtype):
        if not dtype:
            dtype = np.int32
        return dtype(arg.shape[1])
    handle_height_of = handle_heightof

    def handle_depthof(self, arg, dtype):
        if not dtype:
            dtype = np.int32
        return dtype(arg.shape[2])
    handle_depth_of = handle_depthof

    def handle_outlike(self, arg, dtype):
        if not dtype:
            dtype = arg.dtype
        a = np.empty_like(arg, dtype=dtype)
        buf = self.ensure(a)
        self.returns.append((a, buf))
        return buf
    handle_out_like = handle_outlike

    def handle_in(self, arg, dtype):
        arg = arg.astype(dtype, copy=False)
        buf = self.write(arg,wait=not self.async_write)
        return buf

    def handle_inout(self, arg, dtype):
        arg = arg.astype(dtype, copy=False)
        buf = self.write(arg,wait=not self.async_write)
        self.returns.append((arg, buf))
        return buf

    def handle_out(self, arg, dtype):
        arg = arg.astype(dtype, copy=False)
        buf = self.ensure(arg)
        self.returns.append((arg, buf))
        return buf

    def handle_resident(self, arg, dtype):
        arg = arg.astype(dtype, copy=False)
        buf = self.ensure(arg)
        return buf

    def __call__(self, *args: Tuple[Any], **kwargs):
        """Call a kernel, prepare arguments, and track performance"""
        self.kernel_manager.calls += 1
        t = time.time()
        self.global_size = kwargs.pop("global_size", None)
        self.local_size = kwargs.pop("local_size", None)
        assert not isinstance(self.global_size, float) and not isinstance(self.local_size,float), f"{self.global_size},{self.local_size}"
        resident_arrays = kwargs.pop("resident", None)
        prepared_args = self.prepare_arguments(args, resident_arrays=resident_arrays)
        def show_args():
            print("Python arguments")
            for iarg, arg in enumerate(args):
                print(f"Argument: {iarg+1}: {type(arg)} {str(arg)[:64]}")
            print("Kernel arguments")
            for iarg, arg in enumerate(prepared_args):
                print(f"Argument: {iarg+1}: {type(arg)} {str(arg)[:64]}")
        if not isinstance(self.global_size, (tuple, list)):
            self.global_size = (self.global_size,)
        if self.local_size and not isinstance(self.local_size, (tuple, list)):
            self.local_size = (self.local_size,)
        try:
            self.evt = self.kernel(self.kernel_manager.dmgr.queue, self.global_size, self.local_size, *prepared_args)
        except pyopencl.LogicError:
            print(f"Logic error calling {self.name}")
            show_args()
            raise
        except RuntimeError:
            print(f"RuntimeError calling {self.name}")
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
