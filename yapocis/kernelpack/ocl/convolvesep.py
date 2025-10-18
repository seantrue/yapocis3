"""
Auto-generated OpenCL kernel package for convolvesep interface.

Generated from: yapocis/interfaces/convolvesep.interface
Backend: OpenCL
Generation date: 2025-10-17

DO NOT EDIT MANUALLY - Regenerate with:
    yapocis-package convolvesep --backend ocl

This package provides zero-overhead calls to separable convolution kernels
by pre-compiling all buffer management logic.
"""

import numpy as np
from typing import Optional
from yapocis.ocl.device_manager import DeviceManager


class ConvolvesepPackage:
    """
    Compiled kernel package for separable convolution operations.

    Provides zero-overhead calls to convolution kernels by pre-compiling
    all buffer management logic. Supports multiple kernel sizes generated
    from Gaussian weight templates.

    Interface pattern:
        kernel ${name}(int vertical, width_of input, height_of input,
                      in float input, out_like input);
        extension res of ${name}(int vertical, width_of input, height_of input,
                                resident float input, resident float output);
        kernel ${name}_h(width_of input, height_of input,
                        resident float input, resident float output);
        kernel ${name}_v(width_of input, height_of input,
                        resident float input, resident float output);
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

        # Kernel references loaded on demand
        self._kernels: dict[str, any] = {}

    def load(self, kernel_names: list[str]) -> None:
        """
        Load specific kernels from the program.

        Args:
            kernel_names: List of base kernel names (e.g., ['gauss3', 'gauss5'])
                         Automatically loads all variants (_h, _v)

        Note:
            The '_res' variant uses the same OpenCL kernel as the base kernel,
            just with different Python-level buffer handling. There is no
            separate OpenCL kernel for '_res'.
        """
        for name in kernel_names:
            # Main kernel (also used by _res variant)
            self._kernels[name] = getattr(self.program, name)
            # Horizontal and vertical variants are separate kernels
            self._kernels[f"{name}_h"] = getattr(self.program, f"{name}_h")
            self._kernels[f"{name}_v"] = getattr(self.program, f"{name}_v")

    def gauss(
        self,
        kernel_name: str,
        vertical: int,
        input_array: np.ndarray
    ) -> np.ndarray:
        """
        Execute separable convolution kernel with data copy.

        Interface:
            kernel ${name}(int vertical, width_of input, height_of input,
                          in float input, out_like input);

        Args:
            kernel_name: Name of kernel (e.g., 'gauss3', 'gauss5')
            vertical: 0 for horizontal pass, 1 for vertical pass
            input_array: Input image (will be converted to float32)

        Returns:
            Output array with same shape as input (float32)

        Example:
            pkg = ConvolvesepPackage(dmgr, program)
            pkg.load(['gauss3'])
            result = pkg.gauss('gauss3', vertical=0, input_array=image)
        """
        # Param 0: int vertical (scalar)
        vertical_i32 = np.int32(vertical)

        # Param 1: width_of input (derived from input_array.shape[0])
        width = np.int32(input_array.shape[0])

        # Param 2: height_of input (derived from input_array.shape[1])
        height = np.int32(input_array.shape[1])

        # Param 3: in float input (copy to GPU)
        input_array = input_array.astype(np.float32, copy=False)
        buf_input = self.dmgr.write(input_array, wait=True)

        # Param 4: out_like input (allocate output buffer)
        output = np.empty_like(input_array, dtype=np.float32)
        buf_output = self.dmgr.ensure(output)

        # Execute kernel
        global_size = (input_array.size,) if self.autoflatten else input_array.shape
        kernel = self._kernels[kernel_name]
        kernel(
            self.queue,
            global_size,
            None,  # local_size
            vertical_i32,
            width,
            height,
            buf_input,
            buf_output
        )

        # Read result back to host
        self.dmgr.read(output)
        return output

    def gauss_res(
        self,
        kernel_name: str,
        vertical: int,
        input_array: np.ndarray,
        output_array: np.ndarray
    ) -> None:
        """
        Execute separable convolution with resident buffers (no copy).

        Interface:
            extension res of ${name}(int vertical, width_of input, height_of input,
                                    resident float input, resident float output);

        Args:
            kernel_name: Name of kernel (e.g., 'gauss3', 'gauss5')
            vertical: 0 for horizontal pass, 1 for vertical pass
            input_array: Input image already on GPU
            output_array: Output buffer already on GPU

        Note:
            Both arrays must already be resident on GPU (via write() or ensure()).
            No automatic read() - caller must explicitly read results if needed.

        Example:
            pkg.load(['gauss3'])
            buf_in = dmgr.write(image, wait=True)
            buf_out = dmgr.ensure(np.empty_like(image))
            pkg.gauss_res('gauss3', vertical=0,
                         input_array=image, output_array=output)
            dmgr.read(output)  # Explicit read
        """
        # Param 0: int vertical (scalar)
        vertical_i32 = np.int32(vertical)

        # Param 1: width_of input (derived)
        width = np.int32(input_array.shape[0])

        # Param 2: height_of input (derived)
        height = np.int32(input_array.shape[1])

        # Param 3: resident float input (must already be on GPU)
        buf_input = self.dmgr.buffers.get(id(input_array))
        if buf_input is None:
            raise ValueError(
                f"Input array not resident on GPU. "
                f"Call dmgr.write() or dmgr.ensure() first."
            )

        # Param 4: resident float output (must already be on GPU)
        buf_output = self.dmgr.buffers.get(id(output_array))
        if buf_output is None:
            raise ValueError(
                f"Output array not resident on GPU. "
                f"Call dmgr.write() or dmgr.ensure() first."
            )

        # Execute kernel (uses same kernel as gauss(), but with resident buffers)
        global_size = (input_array.size,) if self.autoflatten else input_array.shape
        kernel = self._kernels[kernel_name]  # Same kernel, different buffer handling
        kernel(
            self.queue,
            global_size,
            None,  # local_size
            vertical_i32,
            width,
            height,
            buf_input,
            buf_output
        )
        # Note: No read() - buffers stay resident

    def gauss_h(
        self,
        kernel_name: str,
        input_array: np.ndarray,
        output_array: np.ndarray
    ) -> None:
        """
        Execute horizontal convolution pass with resident buffers.

        Interface:
            kernel ${name}_h(width_of input, height_of input,
                            resident float input, resident float output);

        Args:
            kernel_name: Name of kernel (e.g., 'gauss3', 'gauss5')
            input_array: Input image already on GPU
            output_array: Output buffer already on GPU

        Note:
            Both arrays must already be resident on GPU.
            No automatic read() - caller must explicitly read results.

        Example:
            # Separable convolution: horizontal then vertical
            pkg.load(['gauss3'])
            dmgr.write(aa, wait=True)
            dmgr.ensure(b)
            pkg.gauss_h('gauss3', input_array=aa, output_array=b)
            pkg.gauss_v('gauss3', input_array=b, output_array=aa)
            dmgr.read(aa)
        """
        # Param 0: width_of input (derived)
        width = np.int32(input_array.shape[0])

        # Param 1: height_of input (derived)
        height = np.int32(input_array.shape[1])

        # Param 2: resident float input
        buf_input = self.dmgr.buffers.get(id(input_array))
        if buf_input is None:
            raise ValueError(f"Input array not resident on GPU")

        # Param 3: resident float output
        buf_output = self.dmgr.buffers.get(id(output_array))
        if buf_output is None:
            raise ValueError(f"Output array not resident on GPU")

        # Execute kernel
        global_size = (input_array.size,) if self.autoflatten else input_array.shape
        kernel = self._kernels[f"{kernel_name}_h"]
        kernel(
            self.queue,
            global_size,
            None,  # local_size
            width,
            height,
            buf_input,
            buf_output
        )

    def gauss_v(
        self,
        kernel_name: str,
        input_array: np.ndarray,
        output_array: np.ndarray
    ) -> None:
        """
        Execute vertical convolution pass with resident buffers.

        Interface:
            kernel ${name}_v(width_of input, height_of input,
                            resident float input, resident float output);

        Args:
            kernel_name: Name of kernel (e.g., 'gauss3', 'gauss5')
            input_array: Input image already on GPU
            output_array: Output buffer already on GPU

        Note:
            Both arrays must already be resident on GPU.
            No automatic read() - caller must explicitly read results.
        """
        # Param 0: width_of input (derived)
        width = np.int32(input_array.shape[0])

        # Param 1: height_of input (derived)
        height = np.int32(input_array.shape[1])

        # Param 2: resident float input
        buf_input = self.dmgr.buffers.get(id(input_array))
        if buf_input is None:
            raise ValueError(f"Input array not resident on GPU")

        # Param 3: resident float output
        buf_output = self.dmgr.buffers.get(id(output_array))
        if buf_output is None:
            raise ValueError(f"Output array not resident on GPU")

        # Execute kernel
        global_size = (input_array.size,) if self.autoflatten else input_array.shape
        kernel = self._kernels[f"{kernel_name}_v"]
        kernel(
            self.queue,
            global_size,
            None,  # local_size
            width,
            height,
            buf_input,
            buf_output
        )


class GaussianKernelWrapper:
    """
    Wrapper that mimics the old Kernel interface for backward compatibility.

    This allows existing Gaussian code to work unchanged:
        g = gaussian_kernel
        g.write(array)
        g.h(input, output)
        g.v(output, input)
        g.read(array)
    """

    def __init__(self, package: ConvolvesepPackage, kernel_name: str, width: int):
        """
        Initialize wrapper.

        Args:
            package: ConvolvesepPackage instance
            kernel_name: Base kernel name (e.g., 'gauss3')
            width: Kernel width (for compatibility)
        """
        self.package = package
        self.kernel_name = kernel_name
        self.width = width
        self.dmgr = package.dmgr

    def __call__(self, input_array: np.ndarray, vertical: int = 0) -> np.ndarray:
        """
        Main kernel call - copies data to/from GPU.

        Args:
            input_array: Input image
            vertical: 0 for horizontal, 1 for vertical

        Returns:
            Convolved output
        """
        return self.package.gauss(self.kernel_name, vertical, input_array)

    def res(self, vertical: int, input_array: np.ndarray, output_array: np.ndarray) -> None:
        """
        Extension: resident buffer version.

        Args:
            vertical: 0 for horizontal, 1 for vertical
            input_array: Input (must be resident on GPU)
            output_array: Output (must be resident on GPU)
        """
        self.package.gauss_res(self.kernel_name, vertical, input_array, output_array)

    def h(self, input_array: np.ndarray, output_array: np.ndarray) -> None:
        """
        Horizontal pass with resident buffers.

        Args:
            input_array: Input (must be resident on GPU)
            output_array: Output (must be resident on GPU)
        """
        self.package.gauss_h(self.kernel_name, input_array, output_array)

    def v(self, input_array: np.ndarray, output_array: np.ndarray) -> None:
        """
        Vertical pass with resident buffers.

        Args:
            input_array: Input (must be resident on GPU)
            output_array: Output (must be resident on GPU)
        """
        self.package.gauss_v(self.kernel_name, input_array, output_array)

    def write(self, array: np.ndarray, wait: bool = True):
        """Transfer array to GPU."""
        return self.dmgr.write(array, wait=wait)

    def read(self, array: np.ndarray):
        """Transfer array from GPU."""
        return self.dmgr.read(array)

    def ensure(self, array: np.ndarray):
        """Ensure array buffer exists on GPU."""
        return self.dmgr.ensure(array)
