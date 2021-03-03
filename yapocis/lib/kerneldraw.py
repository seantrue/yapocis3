"""
Higher level wrappers around the OpenCL graphics kernels.
"""
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from yapocis.lib.graphics import program, coerce_coords, coerce_colors, composite_setup, sprites_setup, \
    setup_texts, to_list
from yapocis.utils.basic import chunker
from yapocis.utils.typing import *


class AbstractKernelDraw:
    COLOR_WIDTH = 1

    def __init__(self, chunk_size: Optional[int]=1000):
        self.chunk_size = chunk_size
        self.alpha = 1.0
        self.color = 1.0
        # Initialize display lists
        self._lines = []
        self._rects = []
        self._circles = []
        self._triangles = []
        self._ellipses = []
        self._sprites = []
        self._draw_strings = []

    def coerce_colors(self, n, color=None, colors=None, **kwargs):
        # a tuple, list, array of COLOR_WIDTH floats
        # a list or tuple of array/list/tuple of COLOR_WIDTH floats
        # an array of nxCOLOR_WIDTH floats
        if color is None and colors is None:
            color = self.color
        if color:
            if self.COLOR_WIDTH > 1:
                color = np.asarray(color, dtype=np.float32)
                colors = np.empty((n, self.COLOR_WIDTH), dtype=np.float32)
            else:
                assert isinstance(color, (float, int))
                colors = np.empty(n, dtype=np.float32)
            colors[:] = color
        else:
            colors = np.asarray(colors, dtype=np.float32)
            if self.COLOR_WIDTH > 1:
                if len(colors.shape) == 1:
                    colors = np.column_stack([colors for i in range(self.COLOR_WIDTH)])
                assert colors.shape == (n, self.COLOR_WIDTH)
            else:
                assert colors.shape[0] == n
        return colors

    def coerce_alphas(self, n, alpha=None, alphas=None, **kwargs):
        alphas = to_list(alpha, alphas, n, self.alpha)
        return alphas

    def prepare_args(self, coords, width, **kwargs):
        coords = coerce_coords(coords, width)
        colors = self.coerce_colors(len(coords), **kwargs)
        alphas = self.coerce_alphas(len(coords), **kwargs)
        return coords, colors, alphas

    # Add item to display list for batch rendering
    def line(self, xyxy, distance=1.0, **kwargs):
        self._lines.append((xyxy, distance, kwargs))

    def rect(self, rect, **kwargs):
        self._rects.append((rect, kwargs))

    def circle(self, location, **kwargs):
        self._circles.append((location, kwargs))

    def triangle(self, triangle, **kwargs):
        self._triangles.append((triangle, kwargs))

    def ellipse(self, ellipse, **kwargs):
        self._ellipses.append((ellipse, kwargs))

    def sprite(self, xy, sprite, **kwargs):
        self._sprites.append((xy, sprite, kwargs))

    def draw_string(self, xy, text, font=None, size=None,  alpha=None, color=None, rotation=None):
        self._draw_strings.append((xy, text, dict(font=font, size=size,  alpha=alpha, color=color, rotation=rotation)))

    def _render_list(self, handler, arglist, columns):
        columns = {}
        for kwargs in arglist[-1]:
            for key, value in kwargs.items():
                key = key + "s"
                columns.setdefault(key, []).append(value)
        handler(*arglist[:-1], **columns)

    def render_display_list(self, handle):
        display_list = getattr(self, f"_{handle}", None)
        if display_list:
            handler = getattr(self, f"{handle}", None)
            assert display_list is not None and handler is not None, handler
            if self.chunk_size:
                for chunk in chunker(display_list,self.chunk_size):
                    arglist = list(zip(*chunk))
                    columns = {}
                    for kwargs in arglist[-1]:
                        for key, value in kwargs.items():
                            key = key + "s"
                            columns.setdefault(key, []).append(value)
                    handler(*arglist[:-1], **columns)
            else:
                arglist = list(zip(*display_list))
                columns = {}
                for kwargs in arglist[-1]:
                    for key, value in kwargs.items():
                        key = key + "s"
                        columns.setdefault(key, []).append(value)
                handler(*arglist[:-1], **columns)
            setattr(self, f"_{handle}", [])

    def render_lines(self):
        self.render_display_list("lines")

    def render_rects(self):
        self.render_display_list("rects")

    def render_circles(self):
        self.render_display_list("circles")

    def render_triangles(self):
        self.render_display_list("triangles")

    def render_ellipses(self):
        self.render_display_list("ellipses")

    def render_sprites(self):
        self.render_display_list("sprites")

    def render_draw_strings(self):
        self.render_display_list("draw_strings")

    def render(self):
        self.render_rects()
        self.render_circles()
        self.render_ellipses()
        self.render_triangles()
        self.render_lines()
        self.render_draw_strings()
        self.render_sprites()

    # Implement these to get things done.
    def lines(self, coords, distance=1.0, **kwargs):
        raise NotImplementedError()

    def rects(self, coords, **kwargs):
        raise NotImplementedError()

    def circles(self, coords, **kwargs):
        raise NotImplementedError()

    def triangles(self, coords, **kwargs):
        raise NotImplementedError()

    def ellipses(self, coords, **kwargs):
        raise NotImplementedError()

    def composite(self, locations, blocks, **kwargs):
        raise NotImplementedError()

    def sprites(self, locations, sprites, **kwargs):
        raise NotImplementedError()

    def draw_strings(self, locations, texts, **kwargs):
        raise NotImplementedError()

    def read(self):
        raise NotImplementedError()


class KernelDraw(AbstractKernelDraw):
    COLOR_WIDTH=1
    def __init__(self, a: np.ndarray, **kwargs):
        super(KernelDraw, self).__init__(**kwargs)
        self.a = a
        program.write(a)

    def prepare_args(self, coords, width, **kwargs):
        if not isinstance(coords, np.ndarray) and not coords:
            raise ValueError("coords are not valid")
        coords = coerce_coords(coords, width)
        colors = coerce_colors(coords, **kwargs)
        return coords, colors

    def lines(self, coords, distance=1.0, **kwargs):
        coords, colors = self.prepare_args(coords, 4, **kwargs)
        program.lines_res(self.a, coords, distance, colors)

    def rects(self, coords, **kwargs):
        coords, colors = self.prepare_args(coords, 4, **kwargs)
        program.rects_res(self.a, coords, colors)

    def circles(self, coords, **kwargs):
        coords, colors = self.prepare_args(coords, 3, **kwargs)
        program.circles_res(self.a, coords, colors)

    def triangles(self, coords, **kwargs):
        coords, colors = self.prepare_args(coords, 6, **kwargs)
        program.triangles_res(self.a, coords, colors)

    def ellipses(self, coords, **kwargs):
        coords, colors = self.prepare_args(coords, 4, **kwargs)
        program.ellipses_res(self.a, coords, colors)

    def composite(self, locations, blocks, **kwargs):
        coords, buffer, pixel_indexes = composite_setup(locations, blocks, **kwargs)
        program.composite_res(self.a, coords, buffer, pixel_indexes)

    def sprites(self, locations, sprites, **kwargs):
        coords, buffer, pixel_indexes = sprites_setup(locations, sprites, **kwargs)
        program.composite_res(self.a, coords, buffer, pixel_indexes)

    def draw_strings(self, locations, texts, **kwargs):
        locations, blocks, alphas = setup_texts(locations, texts, **kwargs)
        coords, buffer, pixel_indexes = composite_setup(locations, blocks, alphas=alphas)
        program.composite_res(self.a, coords, buffer, pixel_indexes)

    def read(self):
        program.read(self.a)
        return self.a
