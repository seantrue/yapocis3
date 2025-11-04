#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
    FigureCanvasBase, FigureManagerBase, GraphicsContextBase, RendererBase)
from matplotlib.figure import Figure
from matplotlib.mathtext import MathTextParser
from PIL import Image as PILImage
from yapocis.lib.typeset import extent_freetype

from yapocis.lib.kerneldraw4 import KernelDraw4
import numpy as np
from yapocis.utils.logging import get_logger
import mapbox_earcut as earcut

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
        self.mathtext_parser = MathTextParser('agg')

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
        # Draw outlines (lines) for all polygons
        for points in polygon_points:
            flat_points = points.ravel()
            for i in range(points.shape[0] - 1):
                line = flat_points[i * 2:i * 2 + 4]
                self.kd.line(line, color=color)

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
                    triangle = vertices[i:i + 3].ravel()
                    self.kd.triangle(triangle, color=rgbFace)

        self.kd.render()

    def draw_image(self, gc, x, y, im):
        y = self.height-y-im.shape[0]
        self.kd.composite([(x,y)],[im])

    def get_text_width_height_descent(self, s, prop, ismath):
        # Handle external LaTeX (not supported)
        if ismath == "TeX":
            raise NotImplementedError("TeX rendering not supported in yapocis backend")

        # Handle matplotlib's math text (LaTeX-style syntax)
        if ismath:
            # Use MathTextParser to get dimensions
            ox, oy, width, height, descent, font_image = \
                self.mathtext_parser.parse(s, self.dpi, prop)
            return width, height, descent

        # Normal text rendering using FreeType
        size = self.points_to_pixels(prop.get_size_in_points())
        xmin, ymin, xmax, ymax = extent_freetype(s, "default", size=size)
        w, h, d = xmax - xmin, ymax, abs(ymin)
        # log.debug(f"get_text_width_height_descent {s} {prop} {xmin,ymin,xmax,ymax} -> {w,h,d}")
        return w, h, d

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        # Debug logging
        log.debug(f"draw_text called: s={repr(s)}, ismath={repr(ismath)}")

        # Handle math text separately
        if ismath and ismath != "TeX":
            # Parse and render math text to an image
            ox, oy, width, height, descent, font_image = \
                self.mathtext_parser.parse(s, self.dpi, prop)

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

                # Adjust y coordinate for text baseline
                # y is already in canvas coords (y=0 at top) and represents the baseline
                # top of text image is (height - descent) pixels above the baseline
                # So: y_top = y - (height - descent)
                y_adjusted = y - height + descent

                # Composite the math text image onto the canvas
                self.kd.composite([(x, y_adjusted)], [rgba_image])

            return

        # Normal text rendering using FreeType
        w, h, d = self.get_text_width_height_descent(s, prop, ismath)
        # log.debug(f"draw_text {gc}, {x}, {y}, {s}, {prop}, {angle}, {ismath}, {mtext}")
        size = self.points_to_pixels(prop.get_size_in_points())
        self.kd.draw_string((x, y), s, color=gc._rgb, rotation=angle, size=size)

    def flipy(self):
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


class GraphicsContextTemplate(GraphicsContextBase):
    ...


def draw_if_interactive():
    ...


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

        if (self._renderer is None or
            self._renderer.dpi != dpi or
            self._renderer.figsize != (figwidth, figheight)):
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

    filetypes = {**FigureCanvasBase.filetypes, 'foo': 'My magic Foo format'}

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
        return 'png'


class FigureManagerTemplate(FigureManagerBase):
    ...


FigureCanvas = FigureCanvasTemplate
FigureManager = FigureManagerTemplate
