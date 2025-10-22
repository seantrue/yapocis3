"""
Auto-generated OpenCL kernel package for operators interface.

Generated from: yapocis/interfaces/operators.interface
Backend: OpenCL
Generation date: 2025-10-17

DO NOT EDIT MANUALLY - Regenerate with:
    yapocis-package operators --backend ocl

This package provides zero-overhead calls to binary operator kernels
by pre-compiling all buffer management logic.
"""

import numpy as np
from typing import Optional
from yapocis.ocl.device_manager import DeviceManager


class OperatorsPackage:
    """
    Compiled kernel package for binary operator operations.

    Provides zero-overhead calls to arithmetic kernels (add, sub, mul, divop)
    by pre-compiling all buffer management logic.

    Interface pattern:
        kernel ${name}(in float input1, in float input2, out_like input1);
        extension res of ${name}(resident float input1, resident float input2, resident float output);
        alias div as divop(in float input1, in float input2, out_like input1);
        alias div_res as divop(resident float input1, resident float input2, resident float output);
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
            kernel_names: List of kernel names (e.g., ['add', 'sub', 'mul', 'divop'])

        Note:
            The '_res' variant and aliases use the same OpenCL kernel,
            just with different Python-level buffer handling.
        """
        for name in kernel_names:
            self._kernels[name] = getattr(self.program, name)

        # Also load alias (div -> divop)
        if 'divop' in kernel_names and 'div' not in self._kernels:
            self._kernels['div'] = self._kernels['divop']

    def operator(
        self,
        kernel_name: str,
        input1: np.ndarray,
        input2: np.ndarray
    ) -> np.ndarray:
        """
        Execute binary operator kernel with data copy.

        Interface:
            kernel ${name}(in float input1, in float input2, out_like input1);

        Args:
            kernel_name: Name of kernel (e.g., 'add', 'sub', 'mul', 'divop', 'div')
            input1: First input array (will be converted to float32)
            input2: Second input array (will be converted to float32)

        Returns:
            Output array with same shape as input1 (float32)

        Example:
            pkg = OperatorsPackage(dmgr, program)
            pkg.load(['add', 'sub'])
            result = pkg.operator('add', array1, array2)
        """
        # Param 0: in float input1 (copy to GPU)
        input1 = input1.astype(np.float32, copy=False)
        buf_input1 = self.dmgr.write(input1, wait=True)

        # Param 1: in float input2 (copy to GPU)
        input2 = input2.astype(np.float32, copy=False)
        buf_input2 = self.dmgr.write(input2, wait=True)

        # Param 2: out_like input1 (allocate output buffer)
        output = np.empty_like(input1, dtype=np.float32)
        buf_output = self.dmgr.ensure(output)

        # Execute kernel
        global_size = (input1.size,) if self.autoflatten else input1.shape
        kernel = self._kernels[kernel_name]
        kernel(
            self.queue,
            global_size,
            None,  # local_size
            buf_input1,
            buf_input2,
            buf_output
        )

        # Read result back to host
        self.dmgr.read(output)
        return output

    def operator_res(
        self,
        kernel_name: str,
        input1: np.ndarray,
        input2: np.ndarray,
        output: np.ndarray
    ) -> None:
        """
        Execute binary operator with resident buffers (no copy).

        Interface:
            extension res of ${name}(resident float input1, resident float input2, resident float output);

        Args:
            kernel_name: Name of kernel (e.g., 'add', 'sub', 'mul', 'divop', 'div')
            input1: First input array already on GPU
            input2: Second input array already on GPU
            output: Output buffer already on GPU

        Note:
            All arrays must already be resident on GPU (via write() or ensure()).
            No automatic read() - caller must explicitly read results if needed.
        """
        # Param 0: resident float input1 (must already be on GPU)
        buf_input1 = self.dmgr.buffers.get(id(input1))
        if buf_input1 is None:
            raise ValueError(
                f"Input1 array not resident on GPU. "
                f"Call dmgr.write() or dmgr.ensure() first."
            )

        # Param 1: resident float input2 (must already be on GPU)
        buf_input2 = self.dmgr.buffers.get(id(input2))
        if buf_input2 is None:
            raise ValueError(
                f"Input2 array not resident on GPU. "
                f"Call dmgr.write() or dmgr.ensure() first."
            )

        # Param 2: resident float output (must already be on GPU)
        buf_output = self.dmgr.buffers.get(id(output))
        if buf_output is None:
            raise ValueError(
                f"Output array not resident on GPU. "
                f"Call dmgr.write() or dmgr.ensure() first."
            )

        # Execute kernel (uses same kernel as operator(), but with resident buffers)
        global_size = (input1.size,) if self.autoflatten else input1.shape
        kernel = self._kernels[kernel_name]
        kernel(
            self.queue,
            global_size,
            None,  # local_size
            buf_input1,
            buf_input2,
            buf_output
        )
        # Note: No read() - buffers stay resident
