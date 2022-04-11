#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
    FigureCanvasBase, FigureManagerBase, GraphicsContextBase, RendererBase)
from matplotlib.figure import Figure
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
        self.kd = KernelDraw4(a)

    def draw_path(self, gc, path, transform, rgbFace=None):
        # log.debug(f"draw_path {path} {rgbFace}")
        color = gc._rgb
        polygons = path.to_polygons(transform=transform, closed_only=False)
        # Build a buffer of flipped points, and a separate list of closed polygons
        polygon_points = []
        closed_polygon_points = []
        for polygon in polygons:
            points = np.empty_like(polygon)
            points[:, 1] = polygon[:, 0]
            points[:, 0] = self.height - polygon[:, 1]
            polygon_points.append(points)
            if (points[0, :] - points[-1, :]).sum() == 0.0:
                # Closed polys have start,end point identical
                closed_polygon_points.append(points)
        for points in polygon_points:
            flat_points = points.ravel()
            for i in range(points.shape[0] - 1):
                line = flat_points[i * 2:i * 2 + 4]
                self.kd.line(line, color=color)
        if closed_polygon_points:
            earcut_points = np.concatenate(closed_polygon_points)
            earcut_lengths  = [len(points) for points in closed_polygon_points]
            earcut_lengths = np.cumsum(earcut_lengths)
            simplices = earcut.triangulate_float32(earcut_points, earcut_lengths)
            vertices = earcut_points[simplices]
            first = 0
            for last in earcut_lengths:
                triangles = vertices[first:last]
                for i in range(0, len(triangles), 3):
                    triangle = vertices[i:i + 3].ravel()
                    if first == 0:
                        color = rgbFace
                    else:
                        color = [0,0,0,0]
                    self.kd.triangle(triangle, color=color)
                first = last

        self.kd.render()

    def draw_image(self, gc, x, y, im):
        y = self.height-y-im.shape[0]
        self.kd.composite([(x,y)],[im])

    def get_text_width_height_descent(self, s, prop, ismath):
        size = self.points_to_pixels(prop.get_size_in_points())
        xmin, ymin, xmax, ymax = extent_freetype(s, "default", size=size)
        w, h, d = xmax - xmin, ymax, abs(ymin)
        # log.debug(f"get_text_width_height_descent {s} {prop} {xmin,ymin,xmax,ymax} -> {w,h,d}")
        return w, h, d

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
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
        return a

    def show(self):
        from yapocis.utils.basic import show_array
        a = self.read()
        show_array(a)
        return a


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

    def draw(self):
        renderer = RendererTemplate(self.figure.dpi,
                                    (self.figure.bbox_inches.bounds[2], self.figure.bbox_inches.bounds[3]))
        self.figure.draw(renderer)
        renderer.render()
        a = renderer.show()
        renderer.kd.snapshot(name="mplbackend")
        return a

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
