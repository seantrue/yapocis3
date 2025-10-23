"""
Auto-generated OpenCL kernel package for graphics4 interface.

Generated from: yapocis/interfaces/graphics4.interface
Backend: OpenCL
Generation date: 2025-10-17

This package provides zero-overhead calls to 2D graphics primitives (RGBA).
"""

import numpy as np
from yapocis.cu.device_manager import DeviceManager


class Graphics4Package:
    """
    Compiled kernel package for 2D graphics primitives with RGBA support.

    Interface pattern:
        kernel lines(width_of a, height_of a, inout float a, width_of lines, in float lines, in float distance, in float rcolor);
        extension res as lines(width_of a, height_of a, resident float a, width_of lines, in float lines, in float distance, in float rcolor);
        ... (rects, circles, triangles, ellipses, composite similar)
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

    def lines(
        self,
        canvas: np.ndarray,
        lines: np.ndarray,
        distance: np.ndarray,
        rcolor: np.ndarray
    ) -> np.ndarray:
        """
        Draw lines on canvas with data copy.

        Args:
            canvas: Canvas array (HxWx4) to draw on
            lines: Line data array
            distance: Distance array
            rcolor: Color array (RGBA)

        Returns:
            Modified canvas
        """
        # Param 0: width_of a
        width = np.int32(canvas.shape[0])

        # Param 1: height_of a
        height = np.int32(canvas.shape[1])

        # Param 2: inout float a
        canvas = canvas.astype(np.float32, copy=False)  # No copy needed - write() handles it
        buf_canvas = self.dmgr.write(canvas, wait=True)

        # Param 3: width_of lines
        width_lines = np.int32(lines.shape[0])

        # Param 4: in float lines
        lines = lines.astype(np.float32, copy=False)
        buf_lines = self.dmgr.write(lines, wait=True)

        # Param 5: in float distance
        distance = distance.astype(np.float32, copy=False)
        buf_distance = self.dmgr.write(distance, wait=True)

        # Param 6: in float rcolor
        rcolor = rcolor.astype(np.float32, copy=False)
        buf_rcolor = self.dmgr.write(rcolor, wait=True)

        # Execute kernel
        global_size = (canvas.size,) if self.autoflatten else canvas.shape
        kernel = self._kernels['lines']
        kernel(
            self.queue,
            global_size,
            None,
            width,
            height,
            buf_canvas,
            width_lines,
            buf_lines,
            buf_distance,
            buf_rcolor
        )

        self.dmgr.read(canvas)
        return canvas

    def lines_res(
        self,
        canvas: np.ndarray,
        lines: np.ndarray,
        distance: np.ndarray,
        rcolor: np.ndarray
    ) -> None:
        """Draw lines with resident buffers."""
        width = np.int32(canvas.shape[0])
        height = np.int32(canvas.shape[1])
        width_lines = np.int32(lines.shape[0])

        buf_canvas = self.dmgr.buffers.get(id(canvas))
        if buf_canvas is None:
            raise ValueError("Canvas not resident on GPU.")

        # Lines, distance, rcolor are 'in' so write them
        lines = lines.astype(np.float32, copy=False)
        buf_lines = self.dmgr.write(lines, wait=True)

        distance = distance.astype(np.float32, copy=False)
        buf_distance = self.dmgr.write(distance, wait=True)

        rcolor = rcolor.astype(np.float32, copy=False)
        buf_rcolor = self.dmgr.write(rcolor, wait=True)

        global_size = (canvas.size,) if self.autoflatten else canvas.shape
        kernel = self._kernels['lines']
        kernel(
            self.queue,
            global_size,
            None,
            width,
            height,
            buf_canvas,
            width_lines,
            buf_lines,
            buf_distance,
            buf_rcolor
        )

    def rects(
        self,
        canvas: np.ndarray,
        lines: np.ndarray,
        rcolor: np.ndarray
    ) -> np.ndarray:
        """Draw rectangles on canvas."""
        width = np.int32(canvas.shape[0])
        height = np.int32(canvas.shape[1])

        canvas = canvas.astype(np.float32, copy=False)
        buf_canvas = self.dmgr.write(canvas, wait=True)

        width_lines = np.int32(lines.shape[0])

        lines = lines.astype(np.float32, copy=False)
        buf_lines = self.dmgr.write(lines, wait=True)

        rcolor = rcolor.astype(np.float32, copy=False)
        buf_rcolor = self.dmgr.write(rcolor, wait=True)

        global_size = (canvas.size,) if self.autoflatten else canvas.shape
        kernel = self._kernels['rects']
        kernel(
            self.queue,
            global_size,
            None,
            width,
            height,
            buf_canvas,
            width_lines,
            buf_lines,
            buf_rcolor
        )

        self.dmgr.read(canvas)
        return canvas

    def circles(
        self,
        canvas: np.ndarray,
        lines: np.ndarray,
        rcolor: np.ndarray
    ) -> np.ndarray:
        """Draw circles on canvas."""
        width = np.int32(canvas.shape[0])
        height = np.int32(canvas.shape[1])

        canvas = canvas.astype(np.float32, copy=False)
        buf_canvas = self.dmgr.write(canvas, wait=True)

        width_lines = np.int32(lines.shape[0])

        lines = lines.astype(np.float32, copy=False)
        buf_lines = self.dmgr.write(lines, wait=True)

        rcolor = rcolor.astype(np.float32, copy=False)
        buf_rcolor = self.dmgr.write(rcolor, wait=True)

        global_size = (canvas.size,) if self.autoflatten else canvas.shape
        kernel = self._kernels['circles']
        kernel(
            self.queue,
            global_size,
            None,
            width,
            height,
            buf_canvas,
            width_lines,
            buf_lines,
            buf_rcolor
        )

        self.dmgr.read(canvas)
        return canvas

    def triangles(
        self,
        canvas: np.ndarray,
        lines: np.ndarray,
        rcolor: np.ndarray
    ) -> np.ndarray:
        """Draw triangles on canvas."""
        width = np.int32(canvas.shape[0])
        height = np.int32(canvas.shape[1])

        canvas = canvas.astype(np.float32, copy=False)
        buf_canvas = self.dmgr.write(canvas, wait=True)

        width_lines = np.int32(lines.shape[0])

        lines = lines.astype(np.float32, copy=False)
        buf_lines = self.dmgr.write(lines, wait=True)

        rcolor = rcolor.astype(np.float32, copy=False)
        buf_rcolor = self.dmgr.write(rcolor, wait=True)

        global_size = (canvas.size,) if self.autoflatten else canvas.shape
        kernel = self._kernels['triangles']
        kernel(
            self.queue,
            global_size,
            None,
            width,
            height,
            buf_canvas,
            width_lines,
            buf_lines,
            buf_rcolor
        )

        self.dmgr.read(canvas)
        return canvas

    def ellipses(
        self,
        canvas: np.ndarray,
        lines: np.ndarray,
        rcolor: np.ndarray
    ) -> np.ndarray:
        """Draw ellipses on canvas."""
        width = np.int32(canvas.shape[0])
        height = np.int32(canvas.shape[1])

        canvas = canvas.astype(np.float32, copy=False)
        buf_canvas = self.dmgr.write(canvas, wait=True)

        width_lines = np.int32(lines.shape[0])

        lines = lines.astype(np.float32, copy=False)
        buf_lines = self.dmgr.write(lines, wait=True)

        rcolor = rcolor.astype(np.float32, copy=False)
        buf_rcolor = self.dmgr.write(rcolor, wait=True)

        global_size = (canvas.size,) if self.autoflatten else canvas.shape
        kernel = self._kernels['ellipses']
        kernel(
            self.queue,
            global_size,
            None,
            width,
            height,
            buf_canvas,
            width_lines,
            buf_lines,
            buf_rcolor
        )

        self.dmgr.read(canvas)
        return canvas

    def composite(
        self,
        canvas: np.ndarray,
        lines: np.ndarray,
        pixels: np.ndarray,
        ipixels: np.ndarray
    ) -> np.ndarray:
        """Composite pixels onto canvas."""
        width = np.int32(canvas.shape[0])
        height = np.int32(canvas.shape[1])

        canvas = canvas.astype(np.float32, copy=False)
        buf_canvas = self.dmgr.write(canvas, wait=True)

        width_lines = np.int32(lines.shape[0])

        lines = lines.astype(np.int32, copy=False)
        buf_lines = self.dmgr.write(lines, wait=True)

        pixels = pixels.astype(np.float32, copy=False)
        buf_pixels = self.dmgr.write(pixels, wait=True)

        ipixels = ipixels.astype(np.int32, copy=False)
        buf_ipixels = self.dmgr.write(ipixels, wait=True)

        width_pixels = np.int32(pixels.shape[0])

        global_size = (canvas.size,) if self.autoflatten else canvas.shape
        kernel = self._kernels['composite']
        kernel(
            self.queue,
            global_size,
            None,
            width,
            height,
            buf_canvas,
            width_lines,
            buf_lines,
            buf_pixels,
            buf_ipixels,
            width_pixels
        )

        self.dmgr.read(canvas)
        return canvas
