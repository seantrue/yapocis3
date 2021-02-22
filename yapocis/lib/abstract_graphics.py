#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.
from typing import Optional

import numpy as np

from yapocis.rpc1.programs import load_program
from yapocis.utils.basic import chunker
from yapocis.utils.typing import Coords


class AbstractGraphicsKernel:
    def __init__(self, kernel_name, color_width, device_name: Optional[str] = None):
        self.gp = load_program(kernel_name, device_name=device_name)
        self.color_width = color_width
        self.color = 1

    def read(self, a):
        return self.gp.read(a)

    def write(self, a):
        return self.gp.write(a)

    def coerce_colors(self, n, color=None, colors=None, **kwargs):
        # a tuple, list, array of COLOR_WIDTH floats
        # a list or tuple of array/list/tuple of COLOR_WIDTH floats
        # an array of nxCOLOR_WIDTH floats
        if color is None and colors is None:
            color = self.color
        if color:
            if self.color_width > 1:
                color = np.asarray(color, dtype=np.float32)
                colors = np.empty((n, self.color_width), dtype=np.float32)
            else:
                assert isinstance(color, (float, int))
                colors = np.empty(n, dtype=np.float32)
            colors[:] = color
        else:
            colors = np.asarray(colors, dtype=np.float32)
            if self.color_width > 1:
                if len(colors.shape) == 1:
                    colors = np.column_stack([colors for _ in range(self.color_width)])
                assert colors.shape == (n, self.color_width)
            else:
                assert colors.shape[0] == n
        return colors

    def coerce_coords(self, coords, width, dtype=np.int32):
        if not hasattr(coords, "dtype"):
            coords = np.array(coords, dtype=dtype)
        assert len(coords.shape) == 2 and coords.shape[1] == width
        return coords

    def render_immediate(self, handle: str, coord_width: int, a: np.ndarray, coords: Coords, *args, **kwargs):
        if not isinstance(coords, (np.ndarray, list, tuple)) or not coords:
            return
        args = self.prepare_args(coords, coord_width, *args, **kwargs)
        args = list(args)
        getattr(self.gp, handle)(a, *args, global_size=a.size / self.color_width)

    def rects(self, a, coords: Coords, **kwargs):
        self.render_immediate("rects", 4, a, coords, **kwargs)

    def circles(self, a, coords: Coords, **kwargs):
        self.render_immediate("circles_in", 3, a, coords, **kwargs)

    def triangles(self, a, coords: Coords, **kwargs):
        self.render_immediate("triangles_in", 6, a, coords, **kwargs)

    def ellipses(self, a, coords: Coords, **kwargs):
        self.render_immediate("ellipses_in", 4, a, coords, **kwargs)

    def lines(self, a, coords: Coords, distance: float = 1.0, **kwargs):
        self.render_immediate("lines", 4, a, coords, distance, **kwargs)

    def composite(self, a, locations, blocks, **kwargs):
        self.gp.composite(a, locations, blocks, **kwargs)

    def sprites(self, a, locations, sprites, **kwargs):
        self.gp.sprites(a, locations, sprites, **kwargs)

    def draw_strings(self, a, locations, texts, **kwargs):
        self.gp.draw_strings(a, locations, texts, **kwargs)

    def rects_in(self, a, coords: Coords, **kwargs):
        self.render_immediate("rects_in", 4, a, coords, **kwargs)

    def circles_in(self, a, coords: Coords, **kwargs):
        self.render_immediate("circles_in", 3, a, coords, **kwargs)

    def triangles_in(self, a, coords: Coords, **kwargs):
        self.render_immediate("triangles_in", 6, a, coords, **kwargs)

    def ellipses_in(self, a, coords: Coords, **kwargs):
        self.render_immediate("ellipses_in", 4, a, coords, **kwargs)

    def lines_in(self, a, coords: Coords, distance: float, **kwargs):
        self.render_immediate("lines_in", 4, a, coords, distance, **kwargs)

    def composite_in(self, a, locations, blocks, **kwargs):
        self.gp.composite_in(a, locations, blocks, **kwargs)

    def sprites(self, a, locations, sprites, **kwargs):
        self.gp.sprites_in(self.a, locations, sprites, **kwargs)

    def draw_strings(self, a, locations, texts, **kwargs):
        self.gp.draw_strings_in(a, locations, texts, **kwargs)

    def prepare_args(self, coords, coord_width, **kwargs):
        raise NotImplementedError("prepare_args is not implemented")

    def setup_composite(self, locations, blocks, **kwargs):
        raise NotImplementedError("setup_composite is not implemented")

    def setup_sprites(self, locations, sprites, **kwargs):
        raise NotImplementedError("setup_composite is not implemented")

    def setup_texts(self, locations, texts, font=None, fonts=None, size=None, sizes=None, color=None,
                    colors=None, rotation=None, rotations=None, **kwargs):
        raise NotImplementedError("setup_text is not implemented")


class AbstractKernelDraw:

    def __init__(self, color_width: int = 1, chunk_size: Optional[int] = 1000):
        self.color_width = color_width
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

    def render_immediate(self, handle: str, coord_width: int, coords: Coords, *args, **kwargs):
        if not isinstance(coords, np.ndarray) and not coords:
            return
        getattr(self.gp, handle)(self.a, coords, *args, **kwargs)

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

    def draw_string(self, xy, text, **kwargs):
        self._draw_strings.append((xy, text, kwargs))

    @staticmethod
    def _render_list(handler, arglist, columns):
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
                for chunk in chunker(display_list, self.chunk_size):
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


def to_list(scalar, vector, n, default=None):
    if scalar is None and vector is None:
        scalar = default
    coerced = list(vector) if isinstance(vector, (tuple, list, np.ndarray)) else [scalar] * n
    return coerced


def to_vector(scalar, vector, n, dtype=np.float32, default=None):
    if scalar is None and vector is None:
        scalar = default
    coerced = arrayify1d(scalar, n, dtype=dtype) if scalar else arrayify1d(vector, n, dtype=dtype)
    return coerced


def arrayify1d(scalar_or_vector, length, default=None, dtype=np.float32):
    scalar_or_vector = default if scalar_or_vector is None else scalar_or_vector
    if scalar_or_vector is None:
        raise ValueError(f"Both value and default are None")
    if isinstance(scalar_or_vector, np.ndarray):
        vector = scalar_or_vector.astype(dtype)
    elif isinstance(scalar_or_vector, (list, tuple)):
        vector = np.array(scalar_or_vector, dtype=dtype)
    else:
        vector = [scalar_or_vector] * length
        vector = np.asarray(vector)
    if vector.shape[0] != length:
        raise ValueError(f"value length {vector.shape[0]} does not match specified {length}")
    return vector
