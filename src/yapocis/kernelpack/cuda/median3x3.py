"""
Auto-generated OpenCL kernel package for median3x3 interface.

Generated from: yapocis/interfaces/median3x3.interface
Backend: OpenCL
Generation date: 2025-10-17

This package provides zero-overhead calls to 3x3 median filter kernels.
"""

import numpy as np
from yapocis.cu.device_manager import DeviceManager


class Median3x3Package:
    """
    Compiled kernel package for 3x3 median filter operations.

    Interface pattern:
        kernel median3x3(size_of int a, height_of int a, in float a, out_like a);
        extension res of median3x3(size_of int a, height_of int a, in float a, out float ret);
    """

    def __init__(self, dmgr: DeviceManager, program):
        self.dmgr = dmgr
        self.queue = dmgr.queue
        self.autoflatten = dmgr.autoflatten
        self.program = program
        self._kernel = None

    def load(self) -> None:
        """Load median3x3 kernel from the program."""
        self._kernel = self.program.median3x3

    def median3x3(self, input_array: np.ndarray) -> np.ndarray:
        """
        Execute 3x3 median filter with data copy.

        Args:
            input_array: Input image (will be converted to float32)

        Returns:
            Filtered output array with same shape as input (float32)
        """
        # Param 0: size_of int a (array size in bytes divided by sizeof(int))
        size = np.int32(input_array.size)

        # Param 1: height_of int a (height)
        height = np.int32(input_array.shape[1] if len(input_array.shape) > 1 else 1)

        # Param 2: in float a (copy to GPU)
        input_array = input_array.astype(np.float32, copy=False)
        buf_input = self.dmgr.write(input_array, wait=True)

        # Param 3: out_like a (allocate output buffer)
        output = np.empty_like(input_array, dtype=np.float32)
        buf_output = self.dmgr.ensure(output)

        # Execute kernel
        global_size = (input_array.size,) if self.autoflatten else input_array.shape
        self._kernel(
            self.queue,
            global_size,
            None,  # local_size
            size,
            height,
            buf_input,
            buf_output
        )

        # Read result back to host
        self.dmgr.read(output)
        return output

    def median3x3_res(self, input_array: np.ndarray, output: np.ndarray) -> None:
        """
        Execute 3x3 median filter with resident buffers (no copy).

        Args:
            input_array: Input image already on GPU
            output: Output buffer already on GPU
        """
        # Param 0: size_of int a
        size = np.int32(input_array.size)

        # Param 1: height_of int a
        height = np.int32(input_array.shape[1] if len(input_array.shape) > 1 else 1)

        # Param 2: in float a (must be on GPU - but interface says "in", not "resident")
        buf_input = self.dmgr.write(input_array.astype(np.float32, copy=False), wait=True)

        # Param 3: out float ret
        buf_output = self.dmgr.buffers.get(id(output))
        if buf_output is None:
            raise ValueError("Output array not resident on GPU.")

        # Execute kernel
        global_size = (input_array.size,) if self.autoflatten else input_array.shape
        self._kernel(
            self.queue,
            global_size,
            None,  # local_size
            size,
            height,
            buf_input,
            buf_output
        )
