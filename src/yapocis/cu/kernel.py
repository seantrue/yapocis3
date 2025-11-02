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

    def _package_dimension(self, value, dtype):
        """Package dimension as raw scalar for CUDA."""
        return dtype(value)

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
        return tuple(realargs)

    def handle_outlike(self, arg, dtype):
        if not dtype:
            dtype = arg.dtype
        a = np.empty_like(arg, dtype=dtype)
        buf = self.ensure(a)
        self.returns.append((a, buf))
        return buf

    handle_out_like = handle_outlike

    def _normalize_sizes(self):
        """Normalize global_size and local_size for CUDA."""
        # Check for dispatch_size trait if global_size not explicitly provided
        if self.global_size is None and "dispatch_size" in self.traits:
            from yapocis.core.dispatch_expr import evaluate_dispatch_size
            dispatch_expr = self.traits["dispatch_size"]
            self.global_size = evaluate_dispatch_size(dispatch_expr, self._current_args, self.params)

        # Auto-infer global_size if still None
        if self.global_size is None:
            for arg in self._current_args:
                if hasattr(arg, "size"):
                    if self.autoflatten:
                        self.global_size = arg.size
                    else:
                        self.global_size = arg.shape
                    break

        # Validate sizes
        assert not isinstance(self.global_size, float) and not isinstance(
            self.local_size, float
        ), f"{self.global_size},{self.local_size}"

        # Convert to tuples
        if not isinstance(self.global_size, (tuple, list)):
            self.global_size = (self.global_size,)
        if self.local_size and not isinstance(self.local_size, (tuple, list)):
            self.local_size = (self.local_size,)

        # Calculate grid and block dimensions for CUDA
        self.grid, self.block = self._calculate_grid_block()

    def _execute_kernel_with_args(self, prepared_args, original_args, extra_kwargs):
        """Execute CUDA kernel with prepared arguments."""
        def show_args():
            print("Python arguments")
            for iarg, arg in enumerate(original_args):
                print(f"Argument: {iarg+1}: {type(arg)} {str(arg)[:64]}")
            print("Kernel arguments")
            for iarg, arg in enumerate(prepared_args):
                print(f"Argument: {iarg+1}: {type(arg)} {str(arg)[:64]}")

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
            self.kernel(*cuda_args, block=self.block, grid=self.grid, stream=self.stream)
        except cuda.Error as e:
            print(f"CUDA error calling {self.name}: {e}")
            show_args()
            raise
        except RuntimeError:
            print(f"RuntimeError calling {self.name}")
            show_args()
            raise

    def _handle_async_completion(self):
        """Track CUDA stream for async execution."""
        self.kernel_manager.pending_events.append(self.stream)

    def _handle_sync_completion(self, extra_kwargs):
        """Wait for CUDA stream completion."""
        self.stream.synchronize()

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
