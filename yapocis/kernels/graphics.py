import numpy as np

from yapocis.rpc import kernels
from yapocis.channels import split_channels, join_channels

lines_cl = rects_cl = circles_cl = triangles_cl = ellipses_cl = None
program = kernels.import_program("graphics", lines="lines_cl", rects="rects_cl", circles="circles_cl",
                                 triangles="triangles_cl", ellipses="ellipses_cl")


def coerce_coords(coords, width):
    if not hasattr(coords, "dtype"):
        coords = np.array(coords, dtype=np.float32)
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


class ArrayDraw:
    def __init__(self, a: np.ndarray):
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

    def render(self):
        program.read(self.a)
        return self.a


class ArrayDraw3D:
    def __init__(self, a: np.ndarray):
        self.a = a
        self.color = (None, None, None)
        self.channels = split_channels(a)
        self.ads = [ArrayDraw(a) for a in self.channels]

    def set_color(self, color):
        self.color = color

    def coerce_colors(self, colors):
        if colors is None:
            return self.color
        if isinstance(colors, (list,tuple)):
            colors = np.array(colors)
        if isinstance(colors, np.ndarray):
            if len(colors.shape) == 1:
                colors = tuple(colors[:3])
            elif len(colors.shape) == 2:
                assert colors.shape[1] == 3
                colors = (colors[0,:].flatten(), colors[1,:].flatten(), colors[2:].flatten())
            return colors
        return colors

    def lines(self, coords, distance=1.0, colors=None, alpha=None):
        if not coords:
            return
        colors = self.coerce_colors(colors)
        coords = coerce_coords(coords, 4)
        for ad, rc in zip(self.ads, colors):
            ad.lines(coords, distance=distance, colors=rc, alpha=alpha)

    def rects(self, coords, colors=None, alpha=None):
        if not coords:
            return
        colors = self.coerce_colors(colors)
        coords = coerce_coords(coords, 4)
        for ad, rc in zip(self.ads, colors):
            ad.rects(coords, colors=rc, alpha=alpha)

    def circles(self, coords, colors=None, alpha=None):
        if not coords:
            return
        colors = self.coerce_colors(colors)
        coords = coerce_coords(coords, 3)
        for ad, rc in zip(self.ads, colors):
            ad.circles(coords, colors=rc, alpha=alpha)

    def triangles(self, coords, colors=None, alpha=None):
        if not coords:
            return
        colors = self.coerce_colors(colors)
        coords = coerce_coords(coords, 6)
        for ad, rc in zip(self.ads, colors):
            ad.triangles(coords, colors=rc, alpha=alpha)

    def ellipses(self, coords, colors=None, alpha=None):
        if not coords:
            return
        colors = self.coerce_colors(colors)
        coords = coerce_coords(coords, 4)
        for ad, rc in zip(self.ads, colors):
            ad.ellipses(coords, colors=rc, alpha=alpha)

    def render(self):
        for ad in self.ads:
            ad.render()
        r, g, b = self.channels
        self.a[:, :, 0], self.a[:, :, 1], self.a[:, :, 2] = r.copy(), g.copy(), b.copy()
        return self.a
