from __future__ import annotations

"""
Callable wrapper for PyMetallic kernels, mirroring the OpenCL Kernel wrapper.
"""

import time
from typing import Any, List, Optional, Tuple, Union

import numpy as np

try:
    import pymetallic as pm  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyMetallic is required for the metallic backend") from e

from yapocis.utils.stereo_typing import Array
from yapocis.core.autopad import pad_sizes, infer_wh_from_args


class Kernel:
    def __init__(self, name, kernel_manager: "KernelManager", kernel: "pm.Kernel", params: List[Tuple],
                 async_write: bool = False, local_size: tuple | None = None, tiled: bool = False):
        self.name = name
        self.kernel_manager = kernel_manager
        self.queue = kernel_manager.dmgr.queue
        self.kernel = kernel  # pm.Kernel
        self.params = params
        self.argspecs = self.posbyname = self.returns = self.global_size = None
        self.async_write = async_write
        self.local_size = local_size
        self.tiled = tiled
        self.map_arguments()

    def read(self, a: Array):
        return self.kernel_manager.dmgr.read(a)

    def write(self, a: Array, wait=True):
        return self.kernel_manager.dmgr.write(a, wait=wait)

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

    def prepare_arguments(self, args: List[Any], resident_arrays: Optional[List[str]] = None):
        assert len(self.posbyname) == len(args), f"Got {len(args)}, expected {len(self.posbyname)} arguments, [{self.posbyname}]"
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
                    method_name = "handle_resident"
                real_arg = getattr(self, method_name)(arg, dtype)
            realargs[iparam] = real_arg
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
        buf = self.write(arg, wait=not self.async_write)
        return buf

    def handle_inout(self, arg, dtype):
        arg = arg.astype(dtype, copy=False)
        buf = self.write(arg, wait=not self.async_write)
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
        self.kernel_manager.calls += 1
        t = time.time()
        self.global_size = kwargs.pop("global_size", None)
        self.local_size = kwargs.pop("local_size", None)
        assert not isinstance(self.global_size, float) and not isinstance(self.local_size, float), f"{self.global_size},{self.local_size}"
        resident_arrays = kwargs.pop("resident", None)
        prepared_args = self.prepare_arguments(list(args), resident_arrays=resident_arrays)
        if self.global_size is None:
            if self.tiled:
                self.global_size, self.local_size = pad_sizes(None, self.local_size, args, (16, 16), self.tiled)
            else:
                shape = infer_wh_from_args(args)
                if len(shape) > 1:
                    global_size = (shape[0] * shape[1], 1)
                else:
                    global_size = shape
                self.global_size, self.local_size = global_size, None

        def show_args():
            print("Python arguments")
            for iarg, arg in enumerate(args):
                print(f"Argument: {iarg + 1}: {type(arg)} {str(arg)[:64]}")
            print("Kernel arguments")
            for iarg, arg in enumerate(prepared_args):
                print(f"Argument: {iarg + 1}: {type(arg)} {str(arg)[:64]}")

        # Build buffers and inline-bytes argument lists aligned with [[buffer(i)]]
        buffers: list[tuple[pm.Buffer, int]] = []
        bytes_args: list[tuple[np.ndarray, int]] = []
        for idx, real in enumerate(prepared_args):
            if isinstance(real, pm.Buffer):
                buffers.append((real, idx))
            else:
                arr = real if isinstance(real, np.ndarray) else np.asarray(real)
                bytes_args.append((arr, idx))

        # Convert sizes to 3D grid and optional threadgroup size
        if not isinstance(self.global_size, (tuple, list)):
            grid = (int(self.global_size), 1, 1)
        else:
            if len(self.global_size) == 1:
                grid = (int(self.global_size[0]), 1, 1)
            elif len(self.global_size) == 2:
                grid = (int(self.global_size[0]), int(self.global_size[1]), 1)
            else:
                grid = (int(self.global_size[0]), int(self.global_size[1]), int(self.global_size[2]))
        tgs = None
        if self.local_size is not None:
            ls = self.local_size if isinstance(self.local_size, (tuple, list)) else (self.local_size,)
            if len(ls) == 1:
                tgs = (int(ls[0]), 1, 1)
            elif len(ls) == 2:
                tgs = (int(ls[0]), int(ls[1]), 1)
            else:
                tgs = (int(ls[0]), int(ls[1]), int(ls[2]))

        try:
            self.kernel(self.kernel_manager.dmgr.queue, grid, tgs, buffers=buffers, bytes_args=bytes_args)
        except Exception as e:
            print(f"RuntimeError calling {self.name}: {e}")
            show_args()
            raise
        rval = self.prepare_return()
        self.kernel_manager.runtime += (time.time() - t)
        return rval

    def prepare_return(self) -> Union[Any, Tuple[Any], None]:
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
