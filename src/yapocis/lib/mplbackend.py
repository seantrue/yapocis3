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
from matplotlib.figure import Figure
from matplotlib.mathtext import MathTextParser
from PIL import Image as PILImage

from yapocis.lib.kerneldraw4 import KernelDraw4
from yapocis.lib.typeset import extent_freetype
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
                line = np.array([seg_y1, seg_x1, seg_y2, seg_x2], dtype=np.float32)
                self.kd.line(line, color=color, width=linewidth, aa_width=0.5)
                pos = end_pos
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
        # Get line width from graphics context and convert to pixels
        linewidth = gc.get_linewidth()  # In points
        linewidth_pixels = self.points_to_pixels(linewidth)

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

        # Draw outlines (lines) for all polygons
        for points in polygon_points:
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
                    line = flat_points[i * 2 : i * 2 + 4]
                    line = np.array(
                        [line[1], line[0], line[3], line[2]], dtype=np.float32
                    )
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

                # Render all triangles with the face color
                vertices = earcut_points[simplices]
                for i in range(0, len(vertices), 3):
                    triangle = vertices[i : i + 3].ravel()
                    triangle = np.array(
                        [
                            triangle[1],
                            triangle[0],
                            triangle[3],
                            triangle[2],
                            triangle[5],
                            triangle[4],
                        ],
                        dtype=np.float32,
                    )
                    self.kd.triangle(triangle, color=rgbFace, aa_width=0.5)

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

        # Composite image onto canvas
        # composite expects (row, col) format, so pass (canvas_y, canvas_x)
        self.kd.composite([(canvas_y, canvas_x)], [im])

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

        # Normal text rendering using FreeType
        size = self.points_to_pixels(prop.get_size_in_points())
        xmin, ymin, xmax, ymax = extent_freetype(s, "default", size=size)
        w, h, d = xmax - xmin, ymax, abs(ymin)
        # log.debug(f"get_text_width_height_descent {s} {prop} {xmin,ymin,xmax,ymax} -> {w,h,d}")
        return w, h, d

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):

        # Handle math text separately
        if ismath and ismath != "TeX":
            # Parse and render math text to an image
            ox, oy, width, height, descent, font_image = self.mathtext_parser.parse(
                s, self.dpi, prop
            )

            # Convert grayscale font_image to RGBA with color
            if font_image is not None:
                # Convert FT2Image to numpy array
                font_array = np.asarray(font_image)
                h, w = font_array.shape[:2]
                rgba_image = np.zeros((h, w, 4), dtype=np.float32)

                # Apply text color from graphics context
                color = gc._rgb
                if len(color) == 3:
                    color = list(color) + [1.0]  # Add alpha

                # Set RGB channels to text color
                rgba_image[:, :, 0] = color[0]
                rgba_image[:, :, 1] = color[1]
                rgba_image[:, :, 2] = color[2]

                # Use font_array as alpha channel (normalize to 0-1)
                rgba_image[:, :, 3] = font_array.astype(np.float32) / 255.0

                # Convert from baseline position to top-left for composite
                # y is baseline in canvas coords (y=0 at top)
                # For composite, we need the top-left corner of the image
                # top of text image is (height - descent) pixels above the baseline
                # So: y_top = y - (height - descent)
                y_top = y - (height - descent)

                # Composite the math text image onto the canvas
                # composite expects (row, col) format
                self.kd.composite([(y_top, x)], [rgba_image])

            return

        # Normal text rendering using FreeType
        w, h, d = self.get_text_width_height_descent(s, prop, ismath)
        size = self.points_to_pixels(prop.get_size_in_points())

        # Matplotlib gives us the baseline position (already adjusted for va alignment)
        # Convert from baseline to top-left corner
        # Top of text = baseline - (h - d) where (h - d) is the ascent
        y_top = y - (h - d)

        # Render the text to get the RGBA block
        # Use crop=False to preserve full bounding box for proper baseline alignment
        from yapocis.lib.typeset import render_text

        bbox, text_block = render_text(
            s, font_name="default", size=int(size), rotation=angle, crop=False
        )

        # Apply color to the text block
        rgba_block = np.empty((*text_block.shape, 4), dtype=np.float32)
        rgba_block[:, :, :3] = text_block[:, :, np.newaxis] * np.array(gc._rgb[:3])
        rgba_block[:, :, 3] = text_block * gc._rgb[3]

        # Use composite directly to place text at exact position
        # composite expects locations as (row, col) and handles positioning without offset
        self.kd.composite([(int(np.round(y_top)), int(np.round(x)))], [rgba_block])

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
