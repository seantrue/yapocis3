"""
Auto-generated OpenCL kernel package for gradient interface.

Generated from: yapocis/interfaces/gradient.interface
Backend: OpenCL
Generation date: 2025-10-17

DO NOT EDIT MANUALLY - Regenerate with:
    yapocis-package gradient --backend ocl

This package provides zero-overhead calls to gradient computation kernels
by pre-compiling all buffer management logic.
"""

import numpy as np
from typing import Tuple
from yapocis.cu.device_manager import DeviceManager


class GradientPackage:
    """
    Compiled kernel package for gradient computation operations.

    Provides zero-overhead calls to gradient kernels by pre-compiling
    all buffer management logic.

    Interface pattern:
        kernel gradient(width_of input, height_of input, in float input, int reach,
                       out_like input, out_like input);
        extension res of gradient(width_of input, height_of input, resident float input,
                                 int reach, resident float grad, resident float theta);
    """

    def __init__(self, dmgr: DeviceManager, program):
        """
        Initialize package with device manager and compiled program.

        Args:
            dmgr: OpenCL device manager
            program: Compiled pyopencl.Program containing kernels
        """
        self.dmgr = dmgr
        self.queue = dmgr.queue
        self.autoflatten = dmgr.autoflatten
        self.program = program

        # Kernel references
        self._kernel = None

    def load(self) -> None:
        """Load gradient kernel from the program."""
        self._kernel = self.program.gradient

    def gradient(
        self,
        input_array: np.ndarray,
        reach: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Execute gradient computation kernel with data copy.

        Interface:
            kernel gradient(width_of input, height_of input, in float input, int reach,
                           out_like input, out_like input);

        Args:
            input_array: Input image (will be converted to float32)
            reach: Reach parameter for gradient computation

        Returns:
            Tuple of (grad, theta) arrays, both with same shape as input (float32)

        Example:
            pkg = GradientPackage(dmgr, program)
            pkg.load()
            grad, theta = pkg.gradient(image, reach=3)
        """
        # Param 0: width_of input (derived from input_array.shape[0])
        width = np.int32(input_array.shape[0])

        # Param 1: height_of input (derived from input_array.shape[1])
        height = np.int32(input_array.shape[1])

        # Param 2: in float input (copy to GPU)
        input_array = input_array.astype(np.float32, copy=False)
        buf_input = self.dmgr.write(input_array, wait=True)

        # Param 3: int reach (scalar)
        reach_i32 = np.int32(reach)

        # Param 4: out_like input (allocate grad buffer)
        grad = np.empty_like(input_array, dtype=np.float32)
        buf_grad = self.dmgr.ensure(grad)

        # Param 5: out_like input (allocate theta buffer)
        theta = np.empty_like(input_array, dtype=np.float32)
        buf_theta = self.dmgr.ensure(theta)

        # Execute kernel
        global_size = (input_array.size,) if self.autoflatten else input_array.shape
        self._kernel(
            self.queue,
            global_size,
            None,  # local_size
            width,
            height,
            buf_input,
            reach_i32,
            buf_grad,
            buf_theta
        )

        # Read results back to host
        self.dmgr.read(grad)
        self.dmgr.read(theta)
        return grad, theta

    def gradient_res(
        self,
        input_array: np.ndarray,
        reach: int,
        grad: np.ndarray,
        theta: np.ndarray
    ) -> None:
        """
        Execute gradient computation with resident buffers (no copy).

        Interface:
            extension res of gradient(width_of input, height_of input, resident float input,
                                     int reach, resident float grad, resident float theta);

        Args:
            input_array: Input image already on GPU
            reach: Reach parameter for gradient computation
            grad: Gradient output buffer already on GPU
            theta: Theta output buffer already on GPU

        Note:
            All arrays must already be resident on GPU (via write() or ensure()).
            No automatic read() - caller must explicitly read results if needed.

        Example:
            pkg.load()
            dmgr.write(input_img, wait=True)
            dmgr.ensure(grad_buf)
            dmgr.ensure(theta_buf)
            pkg.gradient_res(input_img, reach=3, grad=grad_buf, theta=theta_buf)
            dmgr.read(grad_buf)
            dmgr.read(theta_buf)
        """
        # Param 0: width_of input (derived)
        width = np.int32(input_array.shape[0])

        # Param 1: height_of input (derived)
        height = np.int32(input_array.shape[1])

        # Param 2: resident float input (must already be on GPU)
        buf_input = self.dmgr.buffers.get(id(input_array))
        if buf_input is None:
            raise ValueError(
                f"Input array not resident on GPU. "
                f"Call dmgr.write() or dmgr.ensure() first."
            )

        # Param 3: int reach (scalar)
        reach_i32 = np.int32(reach)

        # Param 4: resident float grad (must already be on GPU)
        buf_grad = self.dmgr.buffers.get(id(grad))
        if buf_grad is None:
            raise ValueError(
                f"Grad array not resident on GPU. "
                f"Call dmgr.ensure() first."
            )

        # Param 5: resident float theta (must already be on GPU)
        buf_theta = self.dmgr.buffers.get(id(theta))
        if buf_theta is None:
            raise ValueError(
                f"Theta array not resident on GPU. "
                f"Call dmgr.ensure() first."
            )

        # Execute kernel
        global_size = (input_array.size,) if self.autoflatten else input_array.shape
        self._kernel(
            self.queue,
            global_size,
            None,  # local_size
            width,
            height,
            buf_input,
            reach_i32,
            buf_grad,
            buf_theta
        )
        # Note: No read() - buffers stay resident
