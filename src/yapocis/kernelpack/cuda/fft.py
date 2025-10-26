"""
Auto-generated CUDA kernel package for fft interface.

Generated from: yapocis.interfaces.fft
Backend: CUDA
Generation date: 2025-10-25

This package provides zero-overhead calls to GPU kernels.
"""

import numpy as np
from yapocis.cu.device_manager import DeviceManager


class FftPackage:
    """
    Compiled kernel package for fft operations.
    """

    def __init__(self, dmgr: DeviceManager, program):
        self.dmgr = dmgr
        self.queue = dmgr.queue
        self.autoflatten = dmgr.autoflatten
        self.program = program
        self._kernels: dict[str, any] = {}

    def load(self, kernel_names: list[str] | None = None) -> None:
        """Load specified kernels from the program."""
        if kernel_names is None:
            kernel_names = [
                "fft1d",
                "window",
                "power_spectrum",
            ]
        
        for name in kernel_names:
            self._kernels[name] = getattr(self.program, name)

    def fft1d(self, input: np.ndarray, direction: np.int32) -> np.ndarray:
        """
        Execute fft1d kernel.
        """
        # Param 0: in complex64 input
        input = input.astype(np.complex64, copy=False)
        buf_input = self.dmgr.write(input, wait=True)
        # Param 1: out_like input
        output = np.empty_like(input, dtype=np.complex64)
        buf_output = self.dmgr.ensure(output)
        # Param 2: int direction (scalar)
        direction_val = np.int32(direction)

        # Execute kernel
        global_size = (input.size,) if self.autoflatten else input.shape
        self._kernels["fft1d"](
            self.queue,
            global_size,
            None,  # local_size
            buf_input,
            buf_output,
            direction_val,
        )

        # Read results back to host
        self.dmgr.read(output)
        return output

    def window(self, input: np.ndarray, window_type: np.int32) -> np.ndarray:
        """
        Execute window kernel.
        """
        # Param 0: in float input
        input = input.astype(np.float32, copy=False)
        buf_input = self.dmgr.write(input, wait=True)
        # Param 1: int window_type (scalar)
        window_type_val = np.int32(window_type)
        # Param 2: out_like input
        output = np.empty_like(input, dtype=np.float32)
        buf_output = self.dmgr.ensure(output)

        # Execute kernel
        global_size = (input.size,) if self.autoflatten else input.shape
        self._kernels["window"](
            self.queue,
            global_size,
            None,  # local_size
            buf_input,
            window_type_val,
            buf_output,
        )

        # Read results back to host
        self.dmgr.read(output)
        return output

    def power_spectrum(self, fft_output: np.ndarray) -> np.ndarray:
        """
        Execute power_spectrum kernel.
        """
        # Param 0: in complex64 fft_output
        fft_output = fft_output.astype(np.complex64, copy=False)
        buf_fft_output = self.dmgr.write(fft_output, wait=True)
        # Param 1: out_like fft_output
        output = np.empty_like(fft_output, dtype=np.complex64)
        buf_output = self.dmgr.ensure(output)

        # Execute kernel
        global_size = (fft_output.size,) if self.autoflatten else fft_output.shape
        self._kernels["power_spectrum"](
            self.queue,
            global_size,
            None,  # local_size
            buf_fft_output,
            buf_output,
        )

        # Read results back to host
        self.dmgr.read(output)
        return output
