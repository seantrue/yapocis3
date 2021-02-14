#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.
import numpy as np

from yapocis.lib.graphics4 import program, coerce_coords, setup_composite, setup_sprites, setup_texts

from yapocis.lib.kerneldraw import AbstractKernelDraw
from yapocis.utils.typing import *


class KernelDraw4(AbstractKernelDraw):
    COLOR_WIDTH = 4

    def __init__(self, a: np.ndarray, **kwargs):
        super(KernelDraw4, self).__init__(**kwargs)
        self.color = [1.0, 1.0, 1.0, 1.0]
        self.distance = 1.0
        self.alpha = 1.0
        self.a = a
        self.global_size = a.size // 4
        program.write(a)

    def prepare_args(self, coords:Coords, coord_width:int, **kwargs):
        coords = coerce_coords(coords, coord_width)
        colors = self.coerce_colors(len(coords), **kwargs)
        return coords, colors

    def render_immediate(self, handle:str, coord_width:int, coords:Coords, *args, **kwargs):
        if not isinstance(coords, np.ndarray) and not coords:
            return
        coords, colors = self.prepare_args(coords, coord_width, **kwargs)
        getattr(program, handle)(self.a, coords, *args, colors, global_size=self.global_size)


    def rects(self, coords:Coords, **kwargs):
        self.render_immediate("rects_in", 4, coords, **kwargs)

    def circles(self, coords:Coords, **kwargs):
        self.render_immediate("circles_in", 3, coords, **kwargs)

    def triangles(self, coords:Coords, **kwargs):
        self.render_immediate("triangles_in", 6, coords, **kwargs)

    def ellipses(self, coords:Coords, **kwargs):
        self.render_immediate("ellipses_in", 4, coords, **kwargs)

    def lines(self, coords:Coords, distance:float=1.0, **kwargs):
        self.render_immediate("lines_in", 4, coords, distance, **kwargs)

    def render_setup(self, setup:Callable, locations, blocks):
        coords, buffer, pixel_indexes = setup(locations, blocks)
        program.composite_in(self.a, coords, buffer, pixel_indexes, global_size=self.global_size)


    def composite(self, locations, blocks):
        self.render_setup(setup_composite, locations, blocks)

    def sprites(self, locations, sprites):
        self.render_setup(setup_sprites, locations, sprites)

    def draw_strings(self, locations, texts, **kwargs):
        locations, blocks = setup_texts(locations, texts, **kwargs)
        self.composite(locations, blocks)

    def read(self):
        program.read(self.a)
        return self.a
