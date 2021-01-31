import numpy as np

from yapocis.channels import split_channels
from yapocis.lib.graphics import program, coerce_coords, coerce_colors_alpha, composite_setup, sprites_setup, \
    setup_texts


class AbstractKernelDraw:
    def __init__(self):
        # Initialize display lists
        self._lines = []
        self._rects = []
        self._circles = []
        self._triangles = []
        self._ellipses = []
        self._sprites = []
        self._texts = []

    # Add item to display list for batch rendering
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

    # Implement these to get things done.
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
        coords, buffer, pixel_indexes = composite_setup(locations, blocks, alpha)
        program.composite_in(self.a, coords, buffer, pixel_indexes)

    def sprites(self, locations, sprites, alpha=None):
        coords, buffer, pixel_indexes = sprites_setup(locations, sprites, alpha)
        program.composite_in(self.a, coords, buffer, pixel_indexes)

    def draw_strings(self, locations, texts, fonts, sizes, alphas=None):
        locations, blocks, alphas = setup_texts(locations, texts, fonts, sizes, alphas)
        coords, buffer, pixel_indexes = composite_setup(locations, blocks, alphas)
        program.composite_in(self.a, coords, buffer, pixel_indexes)


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