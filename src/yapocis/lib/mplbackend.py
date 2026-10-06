#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

import mapbox_earcut as earcut
import numpy as np
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
    FigureCanvasBase,
    FigureManagerBase,
    GraphicsContextBase,
    RendererBase,
)
from matplotlib.backends.backend_agg import get_hinting_flag
from matplotlib.figure import Figure
from matplotlib.font_manager import findfont, get_font
from matplotlib.mathtext import MathTextParser
from PIL import Image as PILImage

from yapocis.lib.kerneldraw4 import KernelDraw4
from yapocis.utils.logging import get_logger

log = get_logger("mplbackend")


class RendererTemplate(RendererBase):

    def __init__(self, dpi, figsize):
        super().__init__()
        self.dpi = dpi
        self.figsize = figsize
        self.width, self.height = self.get_canvas_width_height()
        a = np.ones((self.height, self.width, 4), dtype=np.float32)
        # Explicitly set alpha channel to 1.0 (opaque white background)
        a[:, :, 3] = 1.0
        self.kd = KernelDraw4(a)
        # Initialize math text parser for rendering LaTeX-style math expressions
        self.mathtext_parser = MathTextParser("agg")

    @staticmethod
    def _kernel_line_width(width_pixels):
        """Width to pass to KernelDraw4.line for a stroke width_pixels wide.

        The lines kernel covers width/2 + 0.5 px on each side of the line, one pixel
        more than width in all; passing width - 1 makes the stroke width_pixels wide,
        as Agg draws it.
        """
        return max(width_pixels - 1.0, 0.0)

    def _draw_dashed_line(
        self, x1, y1, x2, y2, color, linewidth, dash_pattern, dash_offset
    ):
        """Draw a dashed line from (x1,y1) to (x2,y2) using the given dash pattern.

        Args:
            x1, y1: Start point
            x2, y2: End point
            color: Line color [R, G, B, A]
            linewidth: Line width in pixels
            dash_pattern: Tuple/list of (on, off, on, off, ...) lengths in pixels
            dash_offset: Starting offset into the dash pattern (in pixels)
        """
        # Debug
        # log.debug(f"_draw_dashed_line called: pattern={dash_pattern}, offset={dash_offset}")
        # Calculate line length and direction
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx * dx + dy * dy)

        if length < 1e-10:
            return  # Line too short

        # Normalize direction
        ux = dx / length
        uy = dy / length

        # Current position along the line
        pos = 0.0
        # Index into dash pattern
        dash_index = 0
        # Whether currently drawing (True) or skipping (False)
        drawing = True

        # Apply dash offset
        offset_remaining = dash_offset
        while offset_remaining > 0 and dash_index < len(dash_pattern):
            if offset_remaining >= dash_pattern[dash_index]:
                offset_remaining -= dash_pattern[dash_index]
                dash_index = (dash_index + 1) % len(dash_pattern)
                drawing = not drawing
            else:
                pos = offset_remaining
                offset_remaining = 0

        # Draw the dashed line
        while pos < length:
            # Length of current dash/gap segment
            segment_length = dash_pattern[dash_index]

            if drawing:
                # Draw a solid segment
                end_pos = min(pos + segment_length, length)
                seg_x1 = x1 + ux * pos
                seg_y1 = y1 + uy * pos
                seg_x2 = x1 + ux * end_pos
                seg_y2 = y1 + uy * end_pos
                line = np.array([seg_x1, seg_y1, seg_x2, seg_y2], dtype=np.float32)
                self.kd.line(line, color=color, width=linewidth, aa_width=0.5)
                pos = end_pos
            else:
                # Skip a gap
                pos = min(pos + segment_length, length)

            # Move to next dash pattern element
            dash_index = (dash_index + 1) % len(dash_pattern)
            drawing = not drawing

    def draw_path(self, gc, path, transform, rgbFace=None):
        # log.debug(f"draw_path {path} {rgbFace}")
        # Get edge color from graphics context and ensure it has alpha
        color = gc._rgb
        if len(color) == 3:
            color = list(color) + [1.0]  # Add alpha

        # Ensure face color has alpha if provided
        if rgbFace is not None and len(rgbFace) == 3:
            rgbFace = list(rgbFace) + [1.0]  # Add alpha

        polygons = path.to_polygons(transform=transform, closed_only=False)
        # Convert polygon coordinates to canvas space
        polygon_points = []
        closed_polygon_points = []
        for polygon in polygons:
            points = np.empty_like(polygon)
            # Keep X as-is, but flip Y to convert to canvas coordinates (Y=0 at top)
            # This is consistent with draw_image which also flips Y
            points[:, 0] = polygon[:, 0]  # X unchanged
            points[:, 1] = self.height - polygon[:, 1]  # Flip Y
            polygon_points.append(points)
            if (points[0, :] - points[-1, :]).sum() == 0.0:
                # Closed polys have start,end point identical
                closed_polygon_points.append(points)
        # Get line width from graphics context and convert to pixels, as the lines
        # kernel needs it
        linewidth = gc.get_linewidth()  # In points
        linewidth_pixels = self._kernel_line_width(self.points_to_pixels(linewidth))

        # Get line dash pattern from graphics context
        dash_offset, dash_pattern = gc.get_dashes()
        # dash_pattern is None for solid lines, or a tuple of (on, off, on, off, ...)
        # Convert dash pattern from points to pixels
        if dash_pattern is not None:
            dash_pattern_pixels = tuple(self.points_to_pixels(x) for x in dash_pattern)
            dash_offset_pixels = self.points_to_pixels(dash_offset)
        else:
            dash_pattern_pixels = None
            dash_offset_pixels = 0

        # Draw outlines (lines) for all polygons. A zero-width stroke draws nothing, as in
        # Agg; the lines kernel would still cover half a pixel either side.
        for points in polygon_points if linewidth > 0 else []:
            flat_points = points.ravel()
            # If we have a dash pattern, we need to draw dashed lines
            if dash_pattern_pixels:
                # Draw dashed lines segment by segment
                for i in range(points.shape[0] - 1):
                    x1, y1 = points[i]
                    x2, y2 = points[i + 1]
                    self._draw_dashed_line(
                        x1,
                        y1,
                        x2,
                        y2,
                        color,
                        linewidth_pixels,
                        dash_pattern_pixels,
                        dash_offset_pixels,
                    )
            else:
                # Draw solid lines
                for i in range(points.shape[0] - 1):
                    line = np.array(flat_points[i * 2 : i * 2 + 4], dtype=np.float32)
                    self.kd.line(
                        line, color=color, width=linewidth_pixels, aa_width=0.5
                    )

        # Fill closed polygons if face color is provided and visible
        if closed_polygon_points and rgbFace is not None:
            # Check if face color has visible alpha (not fully transparent)
            face_alpha = rgbFace[3] if len(rgbFace) > 3 else 1.0
            if face_alpha > 0.0:
                # Triangulate all closed polygons together
                earcut_points = np.concatenate(closed_polygon_points)
                earcut_lengths = [len(points) for points in closed_polygon_points]
                earcut_lengths = np.cumsum(earcut_lengths)
                simplices = earcut.triangulate_float32(earcut_points, earcut_lengths)

                # Render all triangles with the face color as one mesh, hard-edged.
                # Antialiasing fades every triangle edge inward, so shared edges showed
                # as seams; mesh triangles draw each shared-edge pixel once, so
                # translucent fills are not blended twice along them.
                vertices = earcut_points[simplices]
                for i in range(0, len(vertices), 3):
                    triangle = np.array(vertices[i : i + 3].ravel(), dtype=np.float32)
                    self.kd.mesh_triangle(triangle, color=rgbFace, aa_width=0.0)

        self.kd.render()

    def draw_image(self, gc, x, y, im, transform=None):
        """Draw an image at the specified position.

        Args:
            gc: Graphics context
            x, y: Bottom-left position in display coordinates (pixels)
            im: Image array (RGB or RGBA), already in uint8 [0-255] or float [0-1]
            transform: Optional affine transform (typically not used by matplotlib)

        Notes:
            - matplotlib pre-renders images at the correct size and orientation
            - x, y give the bottom-left corner in matplotlib coordinates (y-up)
            - We need to convert to canvas coordinates (y-down)
            - Image data is already correctly oriented - don't flip it
        """
        # DEBUG: Uncomment to see draw_image calls
        # print(f"draw_image: x={x}, y={y}, im.shape={im.shape}, canvas=({self.height},{self.width})")
        # Convert image to float32 [0, 1] if needed
        if im.dtype == np.uint8:
            im = im.astype(np.float32) / 255.0

        # Ensure RGBA format
        if im.ndim == 2:
            # Grayscale to RGBA
            rgba = np.zeros((*im.shape, 4), dtype=np.float32)
            rgba[:, :, :3] = im[:, :, np.newaxis]
            rgba[:, :, 3] = 1.0
            im = rgba
        elif im.shape[2] == 3:
            # RGB to RGBA
            rgba = np.ones((*im.shape[:2], 4), dtype=np.float32)
            rgba[:, :, :3] = im
            im = rgba

        # Convert matplotlib coordinates to canvas coordinates
        # matplotlib: (x, y) is bottom-left, y increases upward
        # canvas: (x, y) is top-left, y increases downward
        canvas_x = int(x)
        canvas_y = int(self.height - y - im.shape[0])

        # DEBUG: Save image to inspect
        # np.savez(f'/tmp/mplbackend_image_{canvas_x}_{canvas_y}.npz', im=im)

        # Flip image vertically
        # Since flipy() returns True (canvas uses y-down), but matplotlib renders images
        # with row 0 at the top of the image region, we need to flip to match our canvas
        im = np.flipud(im)

        # Composite image onto canvas at (x, y)
        self.kd.composite([(canvas_x, canvas_y)], [im])

    def get_text_width_height_descent(self, s, prop, ismath):
        # Handle external LaTeX (not supported)
        if ismath == "TeX":
            raise NotImplementedError("TeX rendering not supported in yapocis backend")

        # Handle matplotlib's math text (LaTeX-style syntax)
        if ismath:
            # Use MathTextParser to get dimensions
            ox, oy, width, height, descent, font_image = self.mathtext_parser.parse(
                s, self.dpi, prop
            )
            return width, height, descent

        # Normal text: Agg's metrics. Matplotlib lays text out with the advance width and
        # the full height (ascent + descent), not the ink extent.
        font = self._prepare_font(prop)
        font.set_text(s, 0.0, flags=get_hinting_flag())
        w, h = font.get_width_height()
        return w / 64.0, h / 64.0, font.get_descent() / 64.0

    def _prepare_font(self, prop):
        """The FT2Font for prop (the figure's font, not a fixed one) at this dpi."""
        font = get_font(findfont(prop))
        font.clear()
        font.set_size(prop.get_size_in_points(), self.dpi)
        return font

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        # Text is rasterised and placed as Agg does, so output matches the Agg backend:
        # (x, y) is the baseline start, and each branch finds the bitmap's bottom-left
        theta = np.radians(angle)
        if ismath and ismath != "TeX":
            ox, oy, width, height, descent, font_image = self.mathtext_parser.parse(
                s, self.dpi, prop, antialiased=gc.get_antialiased()
            )
            x = round(x + ox + descent * np.sin(theta))
            y = round(y - oy + descent * np.cos(theta))
            image = font_image
        else:
            # The figure's font at the exact (fractional) pixel size, Agg's hinting
            font = self._prepare_font(prop)
            font.set_text(s, 0, flags=get_hinting_flag())
            font.draw_glyphs_to_bitmap(antialiased=gc.get_antialiased())
            d = font.get_descent() / 64.0
            xo, yo = (v / 64.0 for v in font.get_bitmap_offset())
            x = round(x + xo + d * np.sin(theta))
            y = round(y + yo + d * np.cos(theta))
            image = font.get_image()
        self._draw_text_image(image, x, y + 1, angle, gc)

    def _draw_text_image(self, image, x, y, angle, gc):
        """Composite a text coverage bitmap whose bottom-left corner is at (x, y)."""
        if image is None:
            return
        coverage = np.asarray(image).astype(np.float32) / 255.0
        rows, cols = coverage.shape[:2]
        if rows == 0 or cols == 0:
            return
        theta = np.radians(angle)

        if angle % 360 == 0:
            top, left = y - rows, x
        else:
            # Agg rotates the bitmap counterclockwise about its bottom-left corner
            rotated = PILImage.fromarray(coverage, mode="F").rotate(
                angle, resample=PILImage.BILINEAR, expand=True
            )
            # Where the bottom-left corner lands in the expanded, rotated bitmap
            dx, dy = -cols / 2.0, rows / 2.0
            cx, cy = rotated.width / 2.0, rotated.height / 2.0
            corner_x = cx + dx * np.cos(theta) + dy * np.sin(theta)
            corner_y = cy - dx * np.sin(theta) + dy * np.cos(theta)
            top, left = round(y - corner_y), round(x - corner_x)
            coverage = np.asarray(rotated, dtype=np.float32)

        # composite blends straight (not premultiplied) colour by alpha
        rgba_block = np.empty((*coverage.shape, 4), dtype=np.float32)
        rgba_block[:, :, :3] = np.array(gc._rgb[:3])
        rgba_block[:, :, 3] = coverage * gc._rgb[3]
        self.kd.composite([(int(left), int(top))], [rgba_block])

    def flipy(self):
        # Return True to indicate canvas uses y-down coordinate system
        return True

    def get_canvas_width_height(self):
        return int(self.figsize[0] * self.dpi), int(self.figsize[1] * self.dpi)

    def new_gc(self):
        return GraphicsContextTemplate()

    def points_to_pixels(self, points):
        return (points / 72) * self.dpi

    def render(self):
        self.kd.render()

    def read(self):
        self.render()
        a = self.kd.read()
        # Fix alpha channel - KernelDraw4 sets it to 0, but matplotlib expects opaque (1.0)
        # Set alpha to 1.0 for all pixels (opaque canvas)
        a[:, :, 3] = 1.0
        return a

    def show(self):
        from yapocis.utils.basic import show_array

        a = self.read()
        show_array(a)
        return a

    def buffer_rgba(self):
        """Return the rendered buffer as RGBA bytes."""
        a = self.read()
        # Convert float32 [0,1] to uint8 [0,255]
        a = (a * 255).astype(np.uint8)
        return a.tobytes()


class GraphicsContextTemplate(GraphicsContextBase): ...


def draw_if_interactive(): ...


def show(*, block=None):
    for manager in Gcf.get_all_fig_managers():
        # do something to display the GUI
        pass


def new_figure_manager(num, *args, FigureClass=Figure, **kwargs):
    thisFig = FigureClass(*args, **kwargs)
    return new_figure_manager_given_figure(num, thisFig)


def new_figure_manager_given_figure(num, figure):
    canvas = FigureCanvasTemplate(figure)
    manager = FigureManagerTemplate(canvas, num)
    return manager


class FigureCanvasTemplate(FigureCanvasBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._renderer = None

    def get_renderer(self):
        """Get or create the renderer for this canvas."""
        # Check if we need to create or recreate the renderer
        # Recreate if: not exists, or DPI changed, or figure size changed
        figwidth = self.figure.get_figwidth()
        figheight = self.figure.get_figheight()
        dpi = self.figure.dpi

        if (
            self._renderer is None
            or self._renderer.dpi != dpi
            or self._renderer.figsize != (figwidth, figheight)
        ):
            # Use get_figwidth/get_figheight instead of bbox_inches to get the correct figure size
            # bbox_inches might not be finalized until after tight_layout is applied
            self._renderer = RendererTemplate(dpi, (figwidth, figheight))
        return self._renderer

    def draw(self):
        renderer = self.get_renderer()
        self.figure.draw(renderer)
        renderer.render()
        a = renderer.show()
        renderer.kd.snapshot(name="mplbackend")
        return a

    def buffer_rgba(self):
        """Return the rendered buffer as RGBA bytes."""
        renderer = self.get_renderer()
        return renderer.buffer_rgba()

    filetypes = {**FigureCanvasBase.filetypes, "foo": "My magic Foo format"}

    def print_png(self, filename, *args, **kwargs):
        a = self.draw()
        a = a[:, :, :3]
        a = (a * 255).astype(np.uint8)
        # a = a[:,:,:]
        img = PILImage.fromarray(a)
        img.save(filename)

    print_jpg = print_png
    print_jpeg = print_png

    def get_default_filetype(self):
        return "png"


class FigureManagerTemplate(FigureManagerBase): ...


FigureCanvas = FigureCanvasTemplate
FigureManager = FigureManagerTemplate
