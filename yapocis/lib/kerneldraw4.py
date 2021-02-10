#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.
import numpy as np

from yapocis.lib.graphics4 import program, coerce_coords, coerce_colors_alpha, composite_setup, sprites_setup, \
    setup_texts

from yapocis.lib.kerneldraw import AbstractKernelDraw


class KernelDraw4(AbstractKernelDraw):
    def __init__(self, a: np.ndarray):
        super(KernelDraw4, self).__init__()
        self.color = [1.0,1.0,1.0,1.0]
        self.distance = 1.0
        self.alpha = 1.0
        self.a = a
        self.color_width=4
        self.global_size = a.size//4
        program.write(a)


    def prepare_args(self, coords, width, **kwargs):
        coords = coerce_coords(coords, width)
        colors = self.coerce_colors(len(coords), **kwargs)
        return coords, colors

    def render_immediate(self, handle, width, coords, **kwargs):
        if not isinstance(coords, np.ndarray) and not coords:
            return
        coords, colors = self.prepare_args(coords, width, **kwargs)
        colors = coerce_colors_alpha(coords, colors)
        program.rects_in(self.a, coords, colors, global_size=self.global_size)

        for dimension, ad in enumerate(self.ads):
            color_vector = colors[:,dimension]
            getattr(ad, handle)(coords, colors=color_vector, alphas=alphas, distance=kwargs.get("distance", None))

    def lines(self, coords,  distance=1.0, **kwargs):
        if not isinstance(coords, np.ndarray) and not coords:
            return
        coords, colors = self.prepare_args(coords, 4, **kwargs)
        program.lines_in(self.a, coords, distance, colors, global_size=self.global_size)




    def rects(self, coords, **kwargs):
        if not isinstance(coords, np.ndarray) and not coords:
            return
        coords, colors = self.prepare_args(coords, 4, **kwargs)
        program.rects_in(self.a, coords, colors, global_size=self.global_size)

    def circles(self, coords, **kwargs):
        if not isinstance(coords, np.ndarray) and not coords:
            return
        coords, colors = self.prepare_args(coords, 3, **kwargs)
        program.circles_in(self.a, coords, colors, global_size=self.global_size)

    def triangles(self, coords, **kwargs):
        if not isinstance(coords, np.ndarray) and not coords:
            return
        coords, colors = self.prepare_args(coords, 6, **kwargs)
        colors = coerce_colors_alpha(coords, colors)
        program.triangles_in(self.a, coords, colors, global_size=self.global_size)

    def ellipses(self, coords, **kwargs):
        if not isinstance(coords, np.ndarray) and not coords:
            return
        coords, colors = self.prepare_args(coords, 4, **kwargs)
        program.ellipses_in(self.a, coords, colors, global_size=self.global_size)

    def composite(self, locations, blocks):
        coords, buffer, pixel_indexes = composite_setup(locations, blocks)
        program.composite_in(self.a, coords, buffer, pixel_indexes, global_size=self.global_size)

    def sprites(self, locations, sprites):
        coords, buffer, pixel_indexes = sprites_setup(locations, sprites)
        program.composite_in(self.a, coords, buffer, pixel_indexes, global_size=self.global_size)

    def draw_strings(self, locations, texts, **kwargs):
        locations, blocks = setup_texts(locations, texts, **kwargs)
        coords, buffer, pixel_indexes = composite_setup(locations, blocks)
        program.composite_in(self.a, coords, buffer, pixel_indexes, global_size=self.global_size)

    def read(self):
        program.read(self.a)
        return self.a
