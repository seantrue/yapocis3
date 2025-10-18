"""
Auto-generated OpenCL kernel package for zcs interface.

Generated from: yapocis/interfaces/zcs.interface
Backend: OpenCL
Generation date: 2025-10-17

This package provides zero-overhead calls to zero-crossing detection kernels.
"""

import numpy as np
from yapocis.ocl.device_manager import DeviceManager


class ZcsPackage:
    """
    Compiled kernel package for zero-crossing detection operations.

    Interface pattern:
        kernel zcs1(width_of input, height_of input, float threshold, in float input, out_like input);
        alias zcs1_res as zcs1(width_of input, height_of input, float threshold, resident float input, resident float zcs);
        ... (zcs2, zcs3, zcs4 similar)
        kernel sub(in float input1, in float input2, out_like input1);
        alias sub_res as sub(resident float input1, resident float input2, resident float output);
    """

    def __init__(self, dmgr: DeviceManager, program):
        self.dmgr = dmgr
        self.queue = dmgr.queue
        self.autoflatten = dmgr.autoflatten
        self.program = program
        self._kernels: dict[str, any] = {}

    def load(self, kernel_names: list[str]) -> None:
        """Load specific kernels from the program."""
        for name in kernel_names:
            self._kernels[name] = getattr(self.program, name)

    def zcs(self, kernel_name: str, input_array: np.ndarray, threshold: float) -> np.ndarray:
        """
        Execute zero-crossing detection kernel with data copy.

        Args:
            kernel_name: Name of kernel (e.g., 'zcs1', 'zcs2', 'zcs3', 'zcs4')
            input_array: Input image (will be converted to float32)
            threshold: Threshold value for zero-crossing detection

        Returns:
            Zero-crossing map with same shape as input (float32)
        """
        # Param 0: width_of input
        width = np.int32(input_array.shape[0])

        # Param 1: height_of input
        height = np.int32(input_array.shape[1])

        # Param 2: float threshold
        threshold_f32 = np.float32(threshold)

        # Param 3: in float input
        input_array = input_array.astype(np.float32, copy=False)
        buf_input = self.dmgr.write(input_array, wait=True)

        # Param 4: out_like input
        output = np.empty_like(input_array, dtype=np.float32)
        buf_output = self.dmgr.ensure(output)

        # Execute kernel
        global_size = (input_array.size,) if self.autoflatten else input_array.shape
        kernel = self._kernels[kernel_name]
        kernel(
            self.queue,
            global_size,
            None,
            width,
            height,
            threshold_f32,
            buf_input,
            buf_output
        )

        self.dmgr.read(output)
        return output

    def zcs_res(
        self,
        kernel_name: str,
        input_array: np.ndarray,
        threshold: float,
        output: np.ndarray
    ) -> None:
        """
        Execute zero-crossing detection with resident buffers (no copy).

        Args:
            kernel_name: Name of kernel (e.g., 'zcs1', 'zcs2', 'zcs3', 'zcs4')
            input_array: Input image already on GPU
            threshold: Threshold value
            output: Output buffer already on GPU
        """
        width = np.int32(input_array.shape[0])
        height = np.int32(input_array.shape[1])
        threshold_f32 = np.float32(threshold)

        buf_input = self.dmgr.buffers.get(id(input_array))
        if buf_input is None:
            raise ValueError("Input array not resident on GPU.")

        buf_output = self.dmgr.buffers.get(id(output))
        if buf_output is None:
            raise ValueError("Output array not resident on GPU.")

        global_size = (input_array.size,) if self.autoflatten else input_array.shape
        kernel = self._kernels[kernel_name]
        kernel(
            self.queue,
            global_size,
            None,
            width,
            height,
            threshold_f32,
            buf_input,
            buf_output
        )

    def sub(self, input1: np.ndarray, input2: np.ndarray) -> np.ndarray:
        """Execute subtraction kernel with data copy."""
        input1 = input1.astype(np.float32, copy=False)
        input2 = input2.astype(np.float32, copy=False)

        buf_input1 = self.dmgr.write(input1, wait=True)
        buf_input2 = self.dmgr.write(input2, wait=True)

        output = np.empty_like(input1, dtype=np.float32)
        buf_output = self.dmgr.ensure(output)

        global_size = (input1.size,) if self.autoflatten else input1.shape
        kernel = self._kernels['sub']
        kernel(self.queue, global_size, None, buf_input1, buf_input2, buf_output)

        self.dmgr.read(output)
        return output

    def sub_res(self, input1: np.ndarray, input2: np.ndarray, output: np.ndarray) -> None:
        """Execute subtraction with resident buffers."""
        buf_input1 = self.dmgr.buffers.get(id(input1))
        if buf_input1 is None:
            raise ValueError("Input1 array not resident on GPU.")

        buf_input2 = self.dmgr.buffers.get(id(input2))
        if buf_input2 is None:
            raise ValueError("Input2 array not resident on GPU.")

        buf_output = self.dmgr.buffers.get(id(output))
        if buf_output is None:
            raise ValueError("Output array not resident on GPU.")

        global_size = (input1.size,) if self.autoflatten else input1.shape
        kernel = self._kernels['sub']
        kernel(self.queue, global_size, None, buf_input1, buf_input2, buf_output)
