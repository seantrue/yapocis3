#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.
"""
Base class for backend kernel wrappers.

This module provides KernelBase, a shared implementation of common kernel
operations across all GPU backends (OpenCL, Metal, CUDA). The methods in this
class are 100% identical across backends and handle:

- Device memory operations (read/write/ensure)
- Parameter mapping (map_arguments)
- Buffer allocation (handle_in/out/inout/outlike/resident)
- Dimension extraction (widthof/heightof/sizeof with backend-specific packaging)
- Return value preparation (prepare_return)

Backend-specific classes inherit from this base and implement:
- Kernel execution (backend-specific API calls)
- Scalar packaging (_package_dimension) - raw scalar vs 1-element array
- Synchronization (event/stream handling)
"""

from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np

from yapocis.utils.stereo_typing import Array


class KernelBase(ABC):
    """
    Base class for GPU kernel wrappers providing common functionality.

    This class extracts the 100% identical methods that exist across all GPU
    backends (OpenCL, Metal, CUDA) to eliminate code duplication and provide
    a single source of truth for shared kernel logic.

    Backend-specific subclasses must implement:
    - __init__() with backend-specific initialization
    - __call__() with backend-specific kernel execution
    - Dimension handlers (handle_widthof, handle_heightof, etc.)
      with appropriate scalar packaging for their API
    """

    def read(self, a: Array):
        """Read array data from GPU buffer to NumPy array."""
        return self.kernel_manager.dmgr.read(a)

    def write(self, a: Array, wait=True):
        """Write array data from NumPy array to GPU buffer."""
        return self.kernel_manager.dmgr.write(a, wait=wait)

    def ensure(self, a: Array):
        """Ensure array exists as GPU buffer (allocate if needed)."""
        return self.kernel_manager.dmgr.ensure(a)

    def map_arguments(self):
        """
        Build parameter position maps for kernel arguments.

        Creates:
        - self.argspecs: List of all parameter specs (including derived)
        - self.posbyname: Dict mapping parameter names to positions (excluding derived)
        """
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

    def handle_in(self, arg, dtype):
        """
        Handle input parameter: convert to dtype and write to GPU.

        Args:
            arg: NumPy array input
            dtype: Target data type

        Returns:
            GPU buffer handle
        """
        arg = arg.astype(dtype, copy=False)
        buf = self.write(arg, wait=not self.async_write)
        return buf

    def handle_inout(self, arg, dtype):
        """
        Handle input/output parameter: write to GPU and track for return.

        Args:
            arg: NumPy array for input/output
            dtype: Target data type

        Returns:
            GPU buffer handle
        """
        arg = arg.astype(dtype, copy=False)
        buf = self.write(arg, wait=not self.async_write)
        self.returns.append((arg, buf))
        return buf

    def handle_out(self, arg, dtype):
        """
        Handle output parameter: ensure GPU buffer exists and track for return.

        Args:
            arg: NumPy array for output
            dtype: Target data type

        Returns:
            GPU buffer handle
        """
        arg = arg.astype(dtype, copy=False)
        buf = self.ensure(arg)
        self.returns.append((arg, buf))
        return buf

    def handle_resident(self, arg, dtype):
        """
        Handle resident parameter: use existing GPU buffer without transfer.

        Resident buffers are already on the GPU and don't need to be written.
        This avoids redundant data transfers for arrays that persist across calls.

        Args:
            arg: NumPy array with existing GPU buffer
            dtype: Target data type

        Returns:
            GPU buffer handle
        """
        arg = arg.astype(dtype, copy=False)
        buf = self.ensure(arg)
        return buf

    def prepare_return(self) -> Union[Any, tuple[Any], None]:
        """
        Read GPU output buffers back to NumPy arrays and prepare return value.

        Returns:
            - None if no outputs
            - Single array if one output
            - Tuple of arrays if multiple outputs
        """
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

    @abstractmethod
    def _package_dimension(self, value, dtype):
        """
        Package a dimension value for the backend's kernel API.

        Different backends require different scalar packaging:
        - OpenCL/CUDA: Raw scalar (e.g., np.int32(1024))
        - Metal: 1-element array (e.g., np.array([1024], dtype=np.int32))

        Args:
            value: Integer dimension value
            dtype: NumPy dtype for the value

        Returns:
            Backend-appropriate scalar or array
        """
        pass

    def handle_sizeof(self, arg, dtype):
        """
        Extract total array size (number of elements).

        Args:
            arg: NumPy array
            dtype: Target dtype (default: np.int32)

        Returns:
            Packaged size value
        """
        if not dtype:
            dtype = np.int32
        return self._package_dimension(arg.size, dtype)

    handle_size_of = handle_sizeof

    def handle_lengthof(self, arg, dtype):
        """
        Extract array length (len() - typically shape[0] for multi-dimensional).

        Args:
            arg: NumPy array
            dtype: Target dtype (default: np.int32)

        Returns:
            Packaged length value
        """
        if not dtype:
            dtype = np.int32
        return self._package_dimension(len(arg), dtype)

    handle_length_of = handle_lengthof

    def handle_widthof(self, arg, dtype):
        """
        Extract array width (shape[0]).

        Args:
            arg: NumPy array
            dtype: Target dtype (default: np.int32)

        Returns:
            Packaged width value
        """
        if not dtype:
            dtype = np.int32
        return self._package_dimension(arg.shape[0], dtype)

    handle_width_of = handle_widthof

    def handle_heightof(self, arg, dtype):
        """
        Extract array height (shape[1]).

        Args:
            arg: NumPy array
            dtype: Target dtype (default: np.int32)

        Returns:
            Packaged height value
        """
        if not dtype:
            dtype = np.int32
        return self._package_dimension(arg.shape[1], dtype)

    handle_height_of = handle_heightof

    def handle_depthof(self, arg, dtype):
        """
        Extract array depth (shape[2]).

        Args:
            arg: NumPy array
            dtype: Target dtype (default: np.int32)

        Returns:
            Packaged depth value
        """
        if not dtype:
            dtype = np.int32
        return self._package_dimension(arg.shape[2], dtype)

    handle_depth_of = handle_depthof
