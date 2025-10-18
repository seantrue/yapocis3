"""
Auto-generated OpenCL kernel package for anisodiff interface.

Generated from: yapocis/interfaces/anisodiff.interface
Backend: OpenCL
Generation date: 2025-10-17

This package provides zero-overhead calls to anisotropic diffusion kernels.
"""

import numpy as np
from yapocis.ocl.device_manager import DeviceManager


class AnisodiffPackage:
    """
    Compiled kernel package for anisotropic diffusion operations.

    Interface pattern:
        kernel filterImage(width_of input, height_of input, in float input, float scale, float size_step, out_like input);
        alias filterImage_res as filterImage(width_of input, height_of input, resident float input, float scale, float size_step, resident float output);
        ... (filterImage2, filterImage3, filterImage4 similar)
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

    def filterImage(
        self,
        kernel_name: str,
        input_array: np.ndarray,
        scale: float,
        size_step: float
    ) -> np.ndarray:
        """
        Execute anisotropic diffusion filter with data copy.

        Args:
            kernel_name: Name of kernel (e.g., 'filterImage', 'filterImage2', 'filterImage3', 'filterImage4')
            input_array: Input image (will be converted to float32)
            scale: Scale parameter
            size_step: Size step parameter

        Returns:
            Filtered output with same shape as input (float32)
        """
        # Param 0: width_of input
        width = np.int32(input_array.shape[0])

        # Param 1: height_of input
        height = np.int32(input_array.shape[1])

        # Param 2: in float input
        input_array = input_array.astype(np.float32, copy=False)
        buf_input = self.dmgr.write(input_array, wait=True)

        # Param 3: float scale
        scale_f32 = np.float32(scale)

        # Param 4: float size_step
        size_step_f32 = np.float32(size_step)

        # Param 5: out_like input
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
            buf_input,
            scale_f32,
            size_step_f32,
            buf_output
        )

        self.dmgr.read(output)
        return output

    def filterImage_res(
        self,
        kernel_name: str,
        input_array: np.ndarray,
        scale: float,
        size_step: float,
        output: np.ndarray
    ) -> None:
        """
        Execute anisotropic diffusion filter with resident buffers.

        Args:
            kernel_name: Name of kernel
            input_array: Input image already on GPU
            scale: Scale parameter
            size_step: Size step parameter
            output: Output buffer already on GPU
        """
        width = np.int32(input_array.shape[0])
        height = np.int32(input_array.shape[1])
        scale_f32 = np.float32(scale)
        size_step_f32 = np.float32(size_step)

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
            buf_input,
            scale_f32,
            size_step_f32,
            buf_output
        )
