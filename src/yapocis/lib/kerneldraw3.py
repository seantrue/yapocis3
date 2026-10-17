#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.

import numpy as np

from yapocis.lib.abstract_graphics import AbstractKernelDraw
from yapocis.lib.channels import split_planes
from yapocis.lib.kerneldraw1 import KernelDraw1


class KernelDraw3(AbstractKernelDraw):
    COLOR_WIDTH = 3

    def __init__(self, a: np.ndarray, **kwargs):
        super(KernelDraw3, self).__init__(a, **kwargs)
        self.a = a
        self.color = (None, None, None)
        self.alpha = 1.0
        self.channels = [np.ascontiguousarray(pl) for pl in split_planes(a)]
        self.ads = [KernelDraw1(a, **kwargs) for a in self.channels]

    def coerce_coords(self, coords, width, dtype=np.int32, **kwargs):
        if not hasattr(coords, "dtype"):
            coords = np.array(coords, dtype=dtype)
        assert len(coords.shape) == 2 and coords.shape[1] == width
        return coords

    def set_color(self, color, alpha=None):
        self.color = color
        self.alpha = self.alpha if alpha is None else self.alpha

    def prepare_args(self, coords, width, **kwargs):
        coords = self.coerce_coords(coords, width)
        colors = self.coerce_colors(len(coords), **kwargs)
        alphas = self.coerce_alphas(len(coords), **kwargs)
        widths = self.coerce_widths(len(coords), **kwargs)
        return coords, colors, alphas, widths

    def render_immediate(self, handle, width, coords, **kwargs):
        if not coords:
            return
        coords, colors, alphas, widths = self.prepare_args(coords, width, **kwargs)
        for dimension, ad in enumerate(self.ads):
            color_vector = colors[:, dimension]
            getattr(ad, handle)(
                coords, colors=color_vector, alphas=alphas, widths=widths
            )

    def lines(self, coords, distance=1.0, **kwargs):
        self.render_immediate("lines", 4, coords, distance=distance, **kwargs)

    def rects(self, coords, **kwargs):
        self.render_immediate("rects", 4, coords, **kwargs)

    def circles(self, coords, **kwargs):
        self.render_immediate("circles", 3, coords, **kwargs)

    def triangles(self, coords, **kwargs):
        self.render_immediate("triangles", 6, coords, **kwargs)

    def mesh_triangles(self, coords, **kwargs):
        self.render_immediate("mesh_triangles", 6, coords, **kwargs)

    def ellipses(self, coords, **kwargs):
        self.render_immediate("ellipses", 4, coords, **kwargs)

    def sprites(self, locations, sprites, alpha=None):
        if not locations:
            return
        # TODO: Split sprites into planes
        for ad in self.ads:
            ad.sprites(locations, sprites, alpha=alpha)

    def draw_strings(self, locations, texts, **kwargs):
        if not locations:
            return

        for ad in self.ads:
            ad.draw_strings(locations, texts, **kwargs)

    def render(self):
        super(KernelDraw3, self).render()
        for ad in self.ads:
            ad.render()

    def read(self):
        for ad in self.ads:
            ad.read()
        for i, channel in enumerate(self.channels):
            self.a[:, :, i] = channel
        return self.a

    def snapshot(self, **kwargs):
        for ad in self.ads:
            ad.snapshot(**kwargs)
