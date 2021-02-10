#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import numpy as np

from yapocis.lib.channels import split_channels
from yapocis.lib.graphics import to_list, coerce_coords
from yapocis.lib.kerneldraw import AbstractKernelDraw, KernelDraw


class KernelDraw3(AbstractKernelDraw):
    def __init__(self, a: np.ndarray):
        super(KernelDraw3, self).__init__()
        self.a = a
        self.color = (None, None, None)
        self.alpha = 1.0
        self.color_width = 3
        self.channels = split_channels(a)
        self.ads = [KernelDraw(a) for a in self.channels]

    def set_color(self, color, alpha=None):
        self.color = color
        self.alpha = self.alpha if alpha is None else self.alpha

    def prepare_args(self, coords, width, **kwargs):
        coords = coerce_coords(coords, width)
        colors = self.coerce_colors(len(coords), **kwargs)
        alphas = self.coerce_alphas(len(coords), **kwargs)
        return coords, colors, alphas

    def render_immediate(self, handle, width, coords, **kwargs):
        if not coords:
            return
        coords, colors, alphas = self.prepare_args(coords, width, **kwargs)
        for dimension, ad in enumerate(self.ads):
            color_vector = colors[:, dimension]
            getattr(ad, handle)(coords, colors=color_vector, alphas=alphas, distance=kwargs.get("distance", None))

    def lines(self, coords, distance=1.0, **kwargs):
        self.render_immediate("lines", 4, coords, distance=distance, **kwargs)

    def rects(self, coords, **kwargs):
        self.render_immediate("rects", 4, coords, **kwargs)

    def circles(self, coords, **kwargs):
        self.render_immediate("circles", 3, coords, **kwargs)

    def triangles(self, coords, **kwargs):
        self.render_immediate("triangles", 6, coords, **kwargs)

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
        r, g, b = self.channels
        self.a[:, :, 0], self.a[:, :, 1], self.a[:, :, 2] = r.copy(), g.copy(), b.copy()
        return self.a
