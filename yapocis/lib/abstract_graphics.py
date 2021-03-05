
import numpy as np
from .bound_kernels import BoundKernels

class AbstractGraphicsSetup:
    CHANNELS: [int] = 1
    def coerce_coords(self, coords, width, dtype=np.int32, **kwargs):
        raise NotImplementedError

    def coerce_colors(self, coords, color=None, colors=None, alpha=None, alphas=None, **kwargs) -> np.ndarray:
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

    def lines(self, image, coords, width=1.0, **kwargs):
        if not coords:
            return
        coords = self.coerce_coords(coords, 4)
        colors = self.coerce_colors(coords, **kwargs)
        self.kmgr.lines(image, coords, width, colors, global_size=image.size // self.CHANNELS)

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
