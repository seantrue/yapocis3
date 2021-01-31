import numpy as np

import yapocis.rpc.programs
from yapocis.rpc import engines
from yapocis.channels import split_channels
from yapocis.lib.text import render_text

lines_cl = rects_cl = circles_cl = triangles_cl = ellipses_cl = composite_cl = None
program = yapocis.rpc.programs.import_program("graphics", lines="lines_cl", rects="rects_cl", circles="circles_cl",
                                              triangles="triangles_cl", ellipses="ellipses_cl", composite="composite_cl")


def coerce_coords(coords, width, dtype=np.float32):
    if not hasattr(coords, "dtype"):
        coords = np.array(coords, dtype=dtype)
    assert len(coords.shape) == 2 and coords.shape[1] == width
    return coords


def arrayify(scalar_or_vector, length):
    scalar_or_vector = 1.0 if scalar_or_vector is None else scalar_or_vector
    if isinstance(scalar_or_vector, np.ndarray):
        vector = scalar_or_vector
    else:
        vector = np.zeros((length,), dtype=np.float32)
        vector[:] = scalar_or_vector
    return vector


def coerce_colors(coords, colors):
    colors = arrayify(colors, coords.shape[0])
    return colors


def coerce_colors_alpha(coords, colors, alpha):
    colors = arrayify(colors, coords.shape[0])
    alpha = arrayify(alpha, coords.shape[0])
    ca = np.column_stack([colors, alpha])
    return ca


def composite_setup(locations, blocks, alpha):
    """
    Compute the information for compositing a set of distinct blocks
    Args:
        coords: list of tuples composed of a location (x,y)
        blocks: list of pixel blocks, one per element of coords
        alpha: list of scalar or vector of opacity

    Returns:
        coords: list of tuples (x,y,block_width,block_height)
        buffer: float32 array of intensity and opacity per point (2xN)
        ipixels: index of start of block in buffer

    """
    alpha = 1.0 if alpha is None else alpha
    alphas = alpha if isinstance(alpha, list) else arrayify(alpha, len(locations))
    assert len(locations) == len(blocks)
    assert len(alphas) == len(blocks)
    pixel_buffer = []
    alpha_buffer = []
    ipixel = 0
    coords = []
    ipixels = []
    for block, location, alpha in zip(blocks, locations, alphas):
        height, width = block.shape
        block = block.flatten()
        x, y = location
        coords.append((x, y, width, height))
        ipixels.append(ipixel)
        ipixel += len(block)
        pixel_buffer.append(block.flatten())
        alpha = 1.0 if alpha is None else alpha
        alpha = arrayify(alpha, len(block))
        alpha_buffer.append(alpha.flatten())
    pixel_buffer = np.concatenate(pixel_buffer)
    alpha_buffer = np.concatenate(alpha_buffer)
    buffer = np.column_stack([pixel_buffer, alpha_buffer])
    coords = coerce_coords(coords, 4, np.int32)
    ipixels = np.array(ipixels, dtype=np.int32)
    return coords, buffer, ipixels


def sprites_setup(locations, sprites, alpha):
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

    alpha = 1.0 if alpha is None else alpha
    alphas = arrayify(alpha, len(sprites))
    assert len(alphas) == len(sprites)
    pixel_buffer = []
    alpha_buffer = []
    ipixel = 0
    ipixels = []
    for sprite, alpha in zip(sprites, alphas):
        sprite = sprite.flatten()
        ipixels.append(ipixel)
        ipixel += len(sprite)
        pixel_buffer.append(sprite.flatten())
        alpha = 1.0 if alpha is None else alpha
        alpha = arrayify(alpha, len(sprite))
        alpha_buffer.append(alpha)
    pixel_buffer = np.concatenate(pixel_buffer)
    alpha_buffer = np.concatenate(alpha_buffer)
    buffer = np.column_stack([pixel_buffer, alpha_buffer])

    coords = []
    for isprite, x, y in locations:
        sprite = sprites[isprite]
        height, width = sprite.shape
        coords.append((x, y, width, height))
        ipixels.append(ipixels[isprite])
    coords = coerce_coords(coords, 4, np.int32)
    ipixels = np.array(ipixels, dtype=np.int32)
    return coords, buffer, ipixels


def lines(image, coords, distance=1.0, colors=None, alpha=None):
    if not coords:
        return
    coords = coerce_coords(coords, 4)
    colors = coerce_colors_alpha(coords, colors, alpha)
    lines_cl(image, coords, distance, colors)


def rects(image, coords, colors=None, alpha=None):
    if not coords:
        return
    coords = coerce_coords(coords, 4)
    colors = coerce_colors_alpha(coords, colors, alpha)
    rects_cl(image, coords, colors)


def circles(image, coords, colors=None, alpha=None):
    if not coords:
        return
    coords = coerce_coords(coords, 3)
    colors = coerce_colors_alpha(coords, colors, alpha)
    circles_cl(image, coords, colors)


def triangles(image, coords, colors=None, alpha=None):
    if not coords:
        return
    coords = coerce_coords(coords, 6)
    colors = coerce_colors_alpha(coords, colors, alpha)
    triangles_cl(image, coords, colors)


def ellipses(image, coords, colors=None, alpha=None):
    coords = coerce_coords(coords, 4)
    colors = coerce_colors_alpha(coords, colors, alpha)
    ellipses_cl(image, coords, colors)


def composite(image, locations, blocks, alpha=None):
    """
    Composite blocks of pixels into target image
    Args:
        image: 2d array of pixels
        coords: a tuple composed of a location (x,y)
        blocks: list of pixel blocks
        alpha: list of scalar or vector of opacity

    Returns: None

    """
    coords, buffer, ipixels = composite_setup(locations, blocks, alpha)
    composite_cl(image, coords, buffer, ipixels)


def sprites(image, locations, sprites, alphas=None):
    """
    Composite sprites into target image
    Args:
        image: 2d array of pixels
        coords: a tuple composed of a location (sprite_index, x,y)
        blocks: list of sprites blocks
        alpha: optional list of scalar or vector of opacity
    Returns: None

    """
    coords, buffer, ipixels = sprites_setup(locations, sprites, alphas)
    composite_cl(image, coords, buffer, ipixels)


def setup_texts(locations, texts, fonts, sizes, alphas):
    sizes = 32 if sizes is None else sizes
    alphas = 1.0 if alphas is None else alphas
    if not isinstance(fonts, (list, tuple)):
        fonts = [fonts] * len(locations)
    if not isinstance(sizes, (list, tuple)):
        sizes = [sizes] * len(locations)
    if isinstance(alphas, str):
        alphas = [alphas] * len(locations)
    blocks = [render_text(text, font_name, size) for text, font_name, size in zip(texts, fonts, sizes)]
    if isinstance(alphas, (list, tuple)):
        new_alphas = []
        for alpha, block in zip(alphas, blocks):
            if alpha == "mask":
                alpha = block
            new_alphas.append(alpha)
        alphas = new_alphas
    return locations, blocks, alphas

def draw_strings(image, locations, texts, fonts="default", sizes=32, alphas=1.0):
    locations, blocks, alphas = setup_texts(locations, texts, fonts, sizes, alphas)
    composite(image, locations, blocks, alphas)


class AbstractKernelDraw:
    def __init__(self):
        self._lines = []
        self._rects = []
        self._circles = []
        self._triangles = []
        self._ellipses = []
        self._sprites = []
        self._texts = []

    def line(self, xyxy, distance=1.0, color=None, alpha=None):
        self._lines.append((xyxy, distance, color, alpha))

    def rect(self, rect, color=None, alpha=None):
        self._rects.append((rect,color,alpha))

    def circle(self, location, color=None, alpha=None):
        self._circles.append((location,color,alpha))

    def triangle(self, triangle, color=None, alpha=None):
        self._triangles.append((triangle,color,alpha))

    def ellipse(self, ellipse, color=None, alpha=None):
        self._ellipses.append((ellipse,color,alpha))

    def sprite(self, sprite, color=None, alpha=None):
        self._sprites.append((sprite,color,alpha))

    def draw_string(self, xy, text, font="default", size=None, alpha=None):
        size = 32 if size is None else size
        alphas = 1.0 if alpha is None else alpha
        self._texts.append((xy, text, font, size, alpha))

    def render_lines(self):
        if self._lines:
            args = zip(*self._lines)
            self.lines(*args)
        self._lines = []

    def render_rects(self):
        if self._rects:
            args = zip(*self._rects)
            self.rects(*args)
        self._rects = []

    def render_circles(self):
        if self._circles:
            args = zip(*self._circles)
            self.circles(*args)
        self._circles = []

    def render_triangles(self):
        if self._triangles:
            args = zip(*self._triangles)
            self.triangles(*args)
        self._triangles = []

    def render_ellipses(self):
        if self._ellipses:
            args = zip(*self._ellipses)
            self.ellipses(*args)
        self._ellipses = []

    def render_sprites(self):
        if self._sprites:
            args = zip(*self._sprites)
            self.sprites(*args)
        self._sprites = []

    def render_texts(self):
        if self._texts:
            args = zip(*self._texts)
            self.draw_strings(*args)
        self._texts = []

    def render(self):
        self.render_rects()
        self.render_circles()
        self.render_ellipses()
        self.render_triangles()
        self.render_lines()
        self.render_texts()
        self.render_sprites()

    def lines(self, coords, distance=1.0, colors=None, alpha=None):
        raise NotImplementedError()

    def rects(self, coords, colors=None, alpha=None):
        raise NotImplementedError()

    def circles(self, coords, colors=None, alpha=None):
        raise NotImplementedError()

    def triangles(self, coords, colors=None, alpha=None):
        raise NotImplementedError()

    def ellipses(self, coords, colors=None, alpha=None):
        raise NotImplementedError()

    def composite(self, locations, blocks, alpha=None):
        raise NotImplementedError()

    def sprites(self, locations, sprites, alpha=None):
        raise NotImplementedError()

    def draw_strings(self, locations, texts, fonts, sizes, alphas=None):
        raise NotImplementedError()

    def read(self):
        raise NotImplementedError()


class KernelDraw(AbstractKernelDraw):
    def __init__(self, a: np.ndarray):
        super(KernelDraw, self).__init__()
        self.a = a
        program.write(a)

    def lines(self, coords, distance=1.0, colors=None, alpha=None):
        if not isinstance(coords, np.ndarray) and not coords:
            return
        coords = coerce_coords(coords, 4)
        colors = coerce_colors_alpha(coords, colors, alpha)
        program.lines_in(self.a, coords, distance, colors)

    def rects(self, coords, colors=None, alpha=None):
        if not isinstance(coords, np.ndarray) and not coords:
            return
        coords = coerce_coords(coords, 4)
        colors = coerce_colors_alpha(coords, colors, alpha)
        program.rects_in(self.a, coords, colors)

    def circles(self, coords, colors=None, alpha=None):
        if not isinstance(coords, np.ndarray) and not coords:
            return
        coords = coerce_coords(coords, 3)
        colors = coerce_colors_alpha(coords, colors, alpha)
        program.circles_in(self.a, coords, colors)

    def triangles(self, coords, colors=None, alpha=None):
        if not isinstance(coords, np.ndarray) and not coords:
            return
        coords = coerce_coords(coords, 6)
        colors = coerce_colors_alpha(coords, colors, alpha)
        program.triangles_in(self.a, coords, colors)

    def ellipses(self, coords, colors=None, alpha=None):
        if not isinstance(coords, np.ndarray) and not coords:
            return
        coords = coerce_coords(coords, 4)
        colors = coerce_colors_alpha(coords, colors, alpha)
        program.ellipses_in(self.a, coords, colors)

    def composite(self, locations, blocks, alpha=None):
        coords, buffer, ipixels = composite_setup(locations, blocks, alpha)
        program.composite_in(self.a, coords, buffer, ipixels)

    def sprites(self, locations, sprites, alpha=None):
        coords, buffer, ipixels = sprites_setup(locations, sprites, alpha)
        program.composite_in(self.a, coords, buffer, ipixels)

    def draw_strings(self, locations, texts, fonts, sizes, alphas=None):
        locations, blocks, alphas = setup_texts(locations, texts, fonts, sizes, alphas)
        coords, buffer, ipixels = composite_setup(locations, blocks, alphas)
        program.composite_in(self.a, coords, buffer, ipixels)


    def read(self):
        program.read(self.a)
        return self.a


class KernelDraw3(AbstractKernelDraw):
    def __init__(self, a: np.ndarray):
        super(KernelDraw3, self).__init__()
        self.a = a
        self.color = (None, None, None)
        self.channels = split_channels(a)
        self.ads = [KernelDraw(a) for a in self.channels]

    def set_color(self, color):
        self.color = color

    def coerce_colors(self, colors):
        if colors is None:
            return self.color
        if isinstance(colors, (list, tuple)):
            if isinstance(colors[0], np.ndarray):
                return colors
            colors = np.array(colors)
        if isinstance(colors, np.ndarray):
            if len(colors.shape) == 1:
                colors = tuple(colors[:3])
            elif len(colors.shape) == 2:
                assert colors.shape[1] == 3
                colors = (colors[:, 0].flatten(), colors[:, 1].flatten(), colors[:, 2].flatten())
            return colors
        return colors

    def prep_args(self, coords, colors, alpha, width=4):
        colors = self.coerce_colors(colors)
        coords = coerce_coords(coords, width)
        return coords, colors, alpha

    def lines(self, coords, distance=1.0, colors=None, alpha=None):
        if not coords:
            return coords
        coords, colors, alpha = self.prep_args(coords, colors, alpha)
        for ad, rc in zip(self.ads, colors):
            ad.lines(coords, distance=distance, colors=rc, alpha=alpha)

    def rects(self, coords, colors=None, alpha=None):
        if not coords:
            return
        coords, colors, alpha = self.prep_args(coords, colors, alpha)
        for ad, rc in zip(self.ads, colors):
            ad.rects(coords, colors=rc, alpha=alpha)

    def circles(self, coords, colors=None, alpha=None):
        if not coords:
            return
        coords, colors, alpha = self.prep_args(coords, colors, alpha, width=3)
        for ad, rc in zip(self.ads, colors):
            ad.circles(coords, colors=rc, alpha=alpha)

    def triangles(self, coords, colors=None, alpha=None):
        if not coords:
            return
        coords, colors, alpha = self.prep_args(coords, colors, alpha, width=6)
        for ad, rc in zip(self.ads, colors):
            ad.triangles(coords, colors=rc, alpha=alpha)

    def ellipses(self, coords, colors=None, alpha=None):
        if not coords:
            return
        coords, colors, alpha = self.prep_args(coords, colors, alpha)
        for ad, rc in zip(self.ads, colors):
            ad.ellipses(coords, colors=rc, alpha=alpha)

    def sprites(self, locations, sprites, alpha=None):
        if not locations:
            return
        # TODO: Split sprites into planes
        for ad in self.ads:
            ad.sprites(locations, sprites, alpha=alpha)

    def draw_strings(self, locations, texts, fonts, sizes, alphas=None):
        if not locations:
            return
        # TODO: Deal with separate colors for planes
        for ad in self.ads:
            ad.draw_strings(locations, texts, fonts, sizes, alphas=None)

    def render(self):
        super(KernelDraw3, self).render()
        for ad in self.ads:
            ad.render()

    def read(self):
        for ad in self.ads:
            ad.read()
        r, g, b = self.channels
        self.a[:, :, 0], self.a[:, :, 1], self.a[:, :, 2] = r.copy(), g.copy(), b.copy()
        return self.a