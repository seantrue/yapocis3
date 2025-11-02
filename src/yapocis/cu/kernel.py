#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.

import time
from typing import Any, Callable, Union

import numpy as np
import pycuda.driver as cuda

from yapocis.core.kernel_base import KernelBase
from yapocis.utils.stereo_typing import Array

class Kernel(KernelBase):
    """
    Provide a callable interface to a CUDA kernel function.
    """

    def __init__(
        self,
        name,
        kernel_manager: "KernelManager",
        kernel: Callable,
        params: list[tuple],
        async_write: bool = False,
        traits: dict | None = None,
    ):
        self.name = name
        self.kernel_manager = kernel_manager
        self.stream = kernel_manager.dmgr.queue
        self.kernel = kernel
        self.params = params
        self.traits = traits or {}
        self.autoflatten = kernel_manager.dmgr.autoflatten
        self.argspecs = self.posbyname = self.returns = self.local_size = (
            self.global_size
        ) = None
        self.async_write = async_write
        self.map_arguments()

    def prepare_arguments(
        self, args: list[Any], resident_arrays: list[str] | None = None
    ):
        "Takes the actual arguments and maps to the arguments needed by the kernel"
        assert len(self.posbyname) == len(
            args
        ), f" Got {len(args)}, expected {len(self.posbyname)} arguments, [{self.posbyname}]"
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
                    raise RuntimeError(
                        f"Parameter {param.name} is defined as an array, but is a {type(arg)}"
                    )
                if (
                    param.hint != "out_like"
                    and resident_arrays
                    and id(arg) in resident_arrays
                ):
                    method_name = "handle_resident"
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

    def handle_lengthof(self, arg, dtype):
        """Get length of 1D array (size for 1D, shape[0] for multi-dimensional)."""
        if not dtype:
            dtype = np.int32
        # For 1D arrays, length is the size. For multi-D, it's shape[0]
        return dtype(len(arg))

    handle_length_of = handle_lengthof

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

    def __call__(self, *args: tuple[Any], **kwargs):
        """Call a kernel, prepare arguments, and track performance"""
        self.kernel_manager.calls += 1
        t = time.time()
        self.global_size = kwargs.pop("global_size", None)
        self.local_size = kwargs.pop("local_size", None)
        assert not isinstance(self.global_size, float) and not isinstance(
            self.local_size, float
        ), f"{self.global_size},{self.local_size}"

        # Check for async execution mode
        async_exec = kwargs.pop("async_exec", self.kernel_manager._default_async)

        # Check for dispatch_size trait BEFORE prepare_arguments (which auto-sets global_size)
        if self.global_size is None and "dispatch_size" in self.traits:
            from yapocis.core.dispatch_expr import evaluate_dispatch_size
            dispatch_expr = self.traits["dispatch_size"]
            self.global_size = evaluate_dispatch_size(dispatch_expr, args, self.params)

        resident_arrays = kwargs.pop("resident", None)
        prepared_args = self.prepare_arguments(args, resident_arrays=resident_arrays)

        def show_args():
            print("Python arguments")
            for iarg, arg in enumerate(args):
                print(f"Argument: {iarg+1}: {type(arg)} {str(arg)[:64]}")
            print("Kernel arguments")
            for iarg, arg in enumerate(prepared_args):
                print(f"Argument: {iarg+1}: {type(arg)} {str(arg)[:64]}")

        # Convert global_size to grid/block dimensions for CUDA
        if not isinstance(self.global_size, (tuple, list)):
            self.global_size = (self.global_size,)
        if self.local_size and not isinstance(self.local_size, (tuple, list)):
            self.local_size = (self.local_size,)

        # Calculate grid and block dimensions
        grid, block = self._calculate_grid_block()

        # Prepare CUDA kernel arguments - convert GPUArrays to their underlying data
        cuda_args = []
        for arg in prepared_args:
            if hasattr(arg, 'gpudata'):
                # It's a GPUArray, pass the underlying gpudata
                cuda_args.append(arg.gpudata)
            else:
                # It's a scalar or other type
                cuda_args.append(arg)

        try:
            self.kernel(*cuda_args, block=block, grid=grid, stream=self.stream)
        except cuda.Error as e:
            print(f"CUDA error calling {self.name}: {e}")
            show_args()
            raise
        except RuntimeError:
            print(f"RuntimeError calling {self.name}")
            show_args()
            raise

        # Handle synchronization based on async mode
        if async_exec:
            # Async mode: don't synchronize, track stream for later
            self.kernel_manager.pending_events.append(self.stream)
            self.kernel_manager.runtime += time.time() - t
            return None  # Can't return results yet - kernel still executing
        else:
            # Synchronous mode: wait for stream completion (backward compatible)
            poll = kwargs.pop("poll", None)
            if poll:
                # Polling not typically needed with streams, but we can wait
                self.stream.synchronize()
            else:
                self.stream.synchronize()

            rval = self.prepare_return()
            self.kernel_manager.runtime += time.time() - t
            return rval

    def _calculate_grid_block(self):
        """Calculate CUDA grid and block dimensions from global_size and local_size"""
        if self.local_size is None:
            # Default block size
            block = (256, 1, 1)
        else:
            # Pad local_size to 3D
            ls = list(self.local_size) + [1, 1]
            block = tuple(ls[:3])

        # Calculate grid dimensions
        gs = list(self.global_size) + [1, 1]
        gs = gs[:3]

        # Grid = ceil(global / block)
        grid = tuple((gs[i] + block[i] - 1) // block[i] for i in range(3))

        return grid, block
