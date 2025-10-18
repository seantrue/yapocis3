"""
Auto-generated OpenCL kernel package for colormath interface.

Generated from: yapocis/interfaces/colormath.interface
Backend: OpenCL
Generation date: 2025-10-17

This package provides zero-overhead calls to color space transformation kernels.
"""

import numpy as np
from yapocis.cu.device_manager import DeviceManager


class ColormathPackage:
    """
    Compiled kernel package for color space transformation operations.

    Interface pattern:
        kernel transform(in float input, int transform_id, out_like float input);
        alias transform_res as transform(resident float input, int transform_id, resident float output);
        kernel xyz2xyY(in float input, out_like float input);
        alias xyz2xyY_res as xyz2xyY(resident float input, resident float output);
        kernel xyY2xyz(in float input, out_like float input);
        alias xyY2xyz_res as xyY2xyz(resident float input, resident float output);
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

    def transform(self, input_array: np.ndarray, transform_id: int) -> np.ndarray:
        """Execute color transformation with data copy."""
        input_array = input_array.astype(np.float32, copy=False)
        buf_input = self.dmgr.write(input_array, wait=True)

        output = np.empty_like(input_array, dtype=np.float32)
        buf_output = self.dmgr.ensure(output)

        global_size = (input_array.size,) if self.autoflatten else input_array.shape
        kernel = self._kernels['transform']
        kernel(
            self.queue,
            global_size,
            None,
            buf_input,
            np.int32(transform_id),
            buf_output
        )

        self.dmgr.read(output)
        return output

    def transform_res(self, input_array: np.ndarray, transform_id: int, output: np.ndarray) -> None:
        """Execute color transformation with resident buffers."""
        buf_input = self.dmgr.buffers.get(id(input_array))
        if buf_input is None:
            raise ValueError("Input array not resident on GPU.")

        buf_output = self.dmgr.buffers.get(id(output))
        if buf_output is None:
            raise ValueError("Output array not resident on GPU.")

        global_size = (input_array.size,) if self.autoflatten else input_array.shape
        kernel = self._kernels['transform']
        kernel(
            self.queue,
            global_size,
            None,
            buf_input,
            np.int32(transform_id),
            buf_output
        )

    def xyz2xyY(self, input_array: np.ndarray) -> np.ndarray:
        """Execute XYZ to xyY conversion with data copy."""
        input_array = input_array.astype(np.float32, copy=False)
        buf_input = self.dmgr.write(input_array, wait=True)

        output = np.empty_like(input_array, dtype=np.float32)
        buf_output = self.dmgr.ensure(output)

        global_size = (input_array.size,) if self.autoflatten else input_array.shape
        kernel = self._kernels['xyz2xyY']
        kernel(self.queue, global_size, None, buf_input, buf_output)

        self.dmgr.read(output)
        return output

    def xyz2xyY_res(self, input_array: np.ndarray, output: np.ndarray) -> None:
        """Execute XYZ to xyY conversion with resident buffers."""
        buf_input = self.dmgr.buffers.get(id(input_array))
        if buf_input is None:
            raise ValueError("Input array not resident on GPU.")

        buf_output = self.dmgr.buffers.get(id(output))
        if buf_output is None:
            raise ValueError("Output array not resident on GPU.")

        global_size = (input_array.size,) if self.autoflatten else input_array.shape
        kernel = self._kernels['xyz2xyY']
        kernel(self.queue, global_size, None, buf_input, buf_output)

    def xyY2xyz(self, input_array: np.ndarray) -> np.ndarray:
        """Execute xyY to XYZ conversion with data copy."""
        input_array = input_array.astype(np.float32, copy=False)
        buf_input = self.dmgr.write(input_array, wait=True)

        output = np.empty_like(input_array, dtype=np.float32)
        buf_output = self.dmgr.ensure(output)

        global_size = (input_array.size,) if self.autoflatten else input_array.shape
        kernel = self._kernels['xyY2xyz']
        kernel(self.queue, global_size, None, buf_input, buf_output)

        self.dmgr.read(output)
        return output

    def xyY2xyz_res(self, input_array: np.ndarray, output: np.ndarray) -> None:
        """Execute xyY to XYZ conversion with resident buffers."""
        buf_input = self.dmgr.buffers.get(id(input_array))
        if buf_input is None:
            raise ValueError("Input array not resident on GPU.")

        buf_output = self.dmgr.buffers.get(id(output))
        if buf_output is None:
            raise ValueError("Output array not resident on GPU.")

        global_size = (input_array.size,) if self.autoflatten else input_array.shape
        kernel = self._kernels['xyY2xyz']
        kernel(self.queue, global_size, None, buf_input, buf_output)
