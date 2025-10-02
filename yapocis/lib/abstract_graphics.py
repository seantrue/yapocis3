#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

from typing import Optional

import numpy as np

from ..utils.basic import chunker
from .bound_kernels import BoundKernels


class AbstractGraphicsSetup:
    CHANNELS: int = 1
    def coerce_coords(self, coords, width, dtype=np.int32, **kwargs):
        raise NotImplementedError

    def coerce_colors(self, coords, color=None, colors=None, alpha=None, alphas=None, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def coerce_widths(self, coords, width=None, widths=None,  **kwargs) -> np.ndarray:
        raise NotImplementedError


    def setup_composite(self, locations, blocks, alpha=None, alphas=None, **kwargs):
        """
        Compute the information for compositing a set of distinct blocks

        Args:
            locations: list of tuples composed of a location (x,y)
            blocks: list of pixel blocks, one per element of coords
            alpha: list of scalar or vector of opacity

        Returns:
            coords: list of tuples (x,y,block_width,block_height)
            buffer: float32 array of intensity and opacity per point (2xN)
            ipixels: index of start of block in buffer

        """
        raise NotImplementedError


    def setup_sprites(self, locations, sprites, alpha=None, alphas=None, **kwargs):
        """
        Compute the information for compositing a set of reusable blocks, for graphical or repeated elements.

        Args:
            coords: list of tuples composed of a location (isprite, x,y) where isprite is the index into sprites
            sprites: list of pixel blocks, one per element of coords
            alpha: list of scalar or vector of opacity

        Returns:
            coords: list of tuples (x,y,block_width,block_height)
            buffer: float32 array of intensity and opacity per point (2xN)
            ipixels: index of start of block in buffer

        """
        raise NotImplementedError

    def setup_texts(self, locations, texts, font=None, fonts=None, size=None, sizes=None, alpha=None, alphas=None, color=None,
                    colors=None, rotation=None, rotations=None, **kwargs):
        raise NotImplementedError


class AbstractGraphics(BoundKernels):

    def lines(self, image, coords, **kwargs):
        if not coords:
            return
        coords = self.coerce_coords(coords, 4)
        colors = self.coerce_colors(coords, **kwargs)
        widths = self.coerce_widths(coords, **kwargs)
        self.kmgr.lines(image, coords, widths, colors, global_size=image.size // self.CHANNELS)

    def rects(self, image, coords, **kwargs):
        if not coords:
            return
        coords = self.coerce_coords(coords, 4)
        colors = self.coerce_colors(coords, **kwargs)
        self.kmgr.rects(image, coords, colors, global_size=image.size // self.CHANNELS)

    def circles(self, image, coords, **kwargs):
        if not coords:
            return
        coords = self.coerce_coords(coords, 3)
        colors = self.coerce_colors(coords, **kwargs)
        self.kmgr.circles(image, coords, colors, global_size=image.size // self.CHANNELS)

    def triangles(self, image, coords, **kwargs):
        if not coords:
            return
        coords = self.coerce_coords(coords, 6)
        colors = self.coerce_colors(coords, **kwargs)
        self.kmgr.triangles(image, coords, colors, global_size=image.size // self.CHANNELS)

    def ellipses(self, image, coords, **kwargs):
        coords = self.coerce_coords(coords, 4)
        colors = self.coerce_colors(coords, **kwargs)
        self.kmgr.ellipses(image, coords, colors, global_size=image.size // self.CHANNELS)

    def composite(self, image, locations, blocks, **kwargs):
        """
        Composite blocks of pixels into target image

        Args:
            image: 2d array of pixels
            coords: a tuple composed of a location (x,y)
            blocks: list of pixel blocks
            alpha: list of scalar or vector of opacity

        Returns: None

        """
        coords, buffer, ipixels = self.setup_composite(locations, blocks, **kwargs)
        self.kmgr.composite(image, coords, buffer, ipixels, global_size=image.size // self.CHANNELS)

    def sprites(self, image, locations, sprites, alpha=None, alphas=None):
        """
        Composite sprites into target image

        Args:
            image: 2d array of pixels
            coords: a tuple composed of a location (sprite_index, x,y)
            blocks: list of sprites blocks
            alpha: optional list of scalar or vector of opacity
        Returns: None

        """
        coords, buffer, ipixels = self.setup_sprites(locations, sprites, alpha=alpha, alphas=alphas)
        self.kmgr.composite(image, coords, buffer, ipixels, global_size=image.size // self.CHANNELS)

    def draw_strings(self, image, locations, texts, **kwargs):
        locations, blocks, alphas = self.setup_texts(locations, texts, **kwargs)
        self.composite(image, locations, blocks, alphas=alphas)

    def read(self, a):
        return self.kmgr.read(a)

    def write(self, a, async_write=False):
        return self.kmgr.write(a, async_write=async_write)

    def snapshot(self, **kwargs):
        self.kmgr.snapshot(**kwargs)


class AbstractKernelDraw:
    COLOR_WIDTH=None
    GRAPHICS=None

    def __init__(self, a, device_name:Optional[str]=None, graphics:Optional[AbstractGraphics]=None, chunk_size: Optional[int]=500):
        self.a = a
        graphics =self.GRAPHICS if graphics is None else graphics
        self.graphics = graphics(device_name=device_name) if graphics else graphics
        self.chunk_size = chunk_size
        self.global_size = self.a.size//self.COLOR_WIDTH
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
        if self.graphics:
            self.graphics.write(a)

    def prepare_args(self, coords, width, **kwargs):
        coords = self.graphics.coerce_coords(coords, width)
        colors = self.coerce_colors(len(coords), **kwargs)
        widths = self.coerce_widths(len(coords), **kwargs)
        return coords, colors, widths


    def coerce_colors(self, n:int, color:Optional[float]=None, colors=None, alpha=None, **kwargs):
        # a tuple, list, array of COLOR_WIDTH floats
        # a list or tuple of array/list/tuple of COLOR_WIDTH floats
        # an array of nxCOLOR_WIDTH floats
        # For COLOR_WIDTH=1, output is (n, 2) for [color, alpha] pairs
        if color is None and colors is None:
            color = self.color
        if alpha is None:
            alpha = self.alpha
        if color:
            if self.COLOR_WIDTH > 1:
                color = np.asarray(color, dtype=np.float32)
                colors = np.empty((n, self.COLOR_WIDTH), dtype=np.float32)
                colors[:] = color
            else:
                # For single-channel, create (n, 2) array for [color, alpha]
                assert isinstance(color, (float, int))
                colors = np.empty((n, 2), dtype=np.float32)
                colors[:, 0] = color
                colors[:, 1] = alpha
        else:
            colors = np.asarray(colors, dtype=np.float32)
            if self.COLOR_WIDTH > 1:
                if len(colors.shape) == 1:
                    colors = np.column_stack([colors for i in range(self.COLOR_WIDTH)])
                assert colors.shape == (n, self.COLOR_WIDTH)
            else:
                # For single-channel, ensure (n, 2) shape for [color, alpha]
                if len(colors.shape) == 1:
                    colors = np.column_stack([colors, np.full(n, alpha, dtype=np.float32)])
                shape = colors.shape
                assert shape == (n, 2), f"Expected ({n}, 2) but got {shape}"
        return np.ascontiguousarray(colors)

    def coerce_alphas(self, n, alpha=None, alphas=None, **kwargs):
        alphas = to_vector(alpha, alphas, n, default=self.alpha, dtype=np.float32)
        return alphas

    def coerce_widths(self, n, width=None, widths=None, distance=None, **kwargs) -> np.ndarray:
        if width and widths:
            raise RuntimeError(f"Only one of width {width} and {widths}")
        if width and distance:
            raise RuntimeError(f"Only one of width/distance {width} and {distance}")
        width = distance if width is None else width
        widths = to_vector(width, widths, n, default=1.0, dtype=np.float32)
        return widths

    # Add item to display list for batch rendering
    def line(self, xyxy,  **kwargs):
        self._lines.append((xyxy, kwargs))

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

    def lines(self, coords,  **kwargs):
        coords, colors, widths = self.prepare_args(coords, 4, **kwargs)
        self.graphics.kmgr.lines_res(self.a, coords, widths, colors,global_size=self.global_size)

    def rects(self, coords, **kwargs):
        coords, colors, _ = self.prepare_args(coords, 4, **kwargs)
        self.graphics.kmgr.rects_res(self.a, coords, colors,global_size=self.global_size)

    def circles(self, coords, **kwargs):
        coords, colors, _ = self.prepare_args(coords, 3, **kwargs)
        self.graphics.kmgr.circles_res(self.a, coords, colors,global_size=self.global_size)

    def triangles(self, coords, **kwargs):
        coords, colors, _ = self.prepare_args(coords, 6, **kwargs)
        self.graphics.kmgr.triangles_res(self.a, coords, colors,global_size=self.global_size)

    def ellipses(self, coords, **kwargs):
        coords, colors, _ = self.prepare_args(coords, 4, **kwargs)
        self.graphics.kmgr.ellipses_res(self.a, coords, colors,global_size=self.global_size)

    def composite(self, locations, blocks, **kwargs):
        coords, buffer, pixel_indexes = self.graphics.setup_composite(locations, blocks, **kwargs)
        self.graphics.kmgr.composite_res(self.a, coords, buffer, pixel_indexes,global_size=self.global_size)

    def sprites(self, locations, sprites, **kwargs):
        coords, buffer, pixel_indexes = self.graphics.setup_sprites(locations, sprites, **kwargs)
        self.graphics.kmgr.composite_res(self.a, coords, buffer, pixel_indexes,global_size=self.global_size)

    def draw_strings(self, locations, texts, **kwargs):
        locations, blocks, alphas = self.graphics.setup_texts(locations, texts, **kwargs)
        coords, buffer, pixel_indexes = self.graphics.setup_composite(locations, blocks, alphas=alphas)
        self.graphics.kmgr.composite_res(self.a, coords, buffer, pixel_indexes,global_size=self.global_size)

    def read(self):
        self.render()
        self.graphics.read(self.a)
        return self.a

    def snapshot(self, **kwargs):
        self.graphics.snapshot(**kwargs)


def arrayify1d(scalar_or_vector, length, default=None, dtype=np.float32):
    scalar_or_vector = default if scalar_or_vector is None else scalar_or_vector
    if scalar_or_vector is None:
        raise ValueError(f"Both value and default are None")
    if isinstance(scalar_or_vector, np.ndarray):
        vector = scalar_or_vector.astype(dtype, copy=False)
    elif isinstance(scalar_or_vector, (list, tuple)):
        vector = np.array(scalar_or_vector, dtype=dtype)
    else:
        vector = np.empty(length, dtype=dtype)
        vector[:] = scalar_or_vector
    if vector.shape[0] != length:
        raise ValueError(f"value length {vector.shape[0]} does not match specified {length}")
    return vector


def arrayify2d(scalar_or_vector, length, width, default=None, dtype=np.float32):
    if scalar_or_vector is None and default is None:
        raise ValueError(f"Both value and default are None")
    scalar_or_vector = [default] * width if scalar_or_vector is None else scalar_or_vector
    if isinstance(scalar_or_vector, (list, tuple, np.ndarray)):
        vector = np.array(scalar_or_vector, dtype=dtype)
    if len(vector.shape) == 2:
        if vector.shape[2] != width:
            raise ValueError(f"shape {scalar_or_vector.shape} should be {length},{width}")
        return vector
    else:
        vector = np.empty((length, width), dtype=dtype)
        vector[:] = scalar_or_vector
    return vector


def to_list(scalar, vector, n, default=None):
    if scalar is None and vector is None:
        scalar = default
    coerced = list(vector) if isinstance(vector, (tuple, list, np.ndarray)) else [scalar] * n
    coerced = [default if value is None else value for value in coerced]
    assert len(coerced) == n
    return coerced


def to_vector(scalar, vector, n, dtype=np.float32, default=None):
    if scalar is None and vector is None:
        scalar = default
    coerced = arrayify1d(scalar, n, default=default, dtype=dtype) if scalar else arrayify1d(vector, n, default=default,dtype=dtype)
    return coerced