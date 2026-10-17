"""
Low level wrappers around the OpenCL graphics kernels.
"""

#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.

import numpy as np

from yapocis.lib.typeset import compute_offset, render_text
from yapocis.utils.basic import compute_360_to_180

from .abstract_graphics import (
    AbstractGraphics,
    AbstractGraphicsSetup,
    arrayify1d,
    to_list,
    to_vector,
)


class Graphics1Setup(AbstractGraphicsSetup):
    CHANNELS = 1

    def coerce_coords(self, coords, width, dtype=np.int32, **kwargs):
        if not hasattr(coords, "dtype"):
            coords = np.array(coords, dtype=dtype)
        assert len(coords.shape) == 2 and coords.shape[1] == width
        return coords

    def coerce_colors(
        self, coords, color=None, colors=None, alpha=None, alphas=None, **kwargs
    ) -> np.ndarray:
        n = coords.shape[0]
        if color and colors:
            raise RuntimeError(f"Only one of color {color} and {colors}")
        if alpha and alphas:
            raise RuntimeError(f"Only one of color {alpha} and {alphas}")
        colors = to_vector(color, colors, n, default=1.0, dtype=np.float32)
        alphas = to_vector(alpha, alphas, n, default=1.0, dtype=np.float32)
        ca = np.column_stack([colors, alphas])
        return ca

    def coerce_widths(
        self, coords, width=None, widths=None, distance=None, **kwargs
    ) -> np.ndarray:
        n = coords.shape[0]
        if width and widths:
            raise RuntimeError(f"Only one of width {width} and {widths}")
        if width and distance:
            raise RuntimeError(f"Only one of width/distance {width} and {distance}")
        width = distance if width is None else width
        widths = to_vector(width, widths, n, default=1.0, dtype=np.float32)
        return widths

    def coerce_aa_widths(
        self, coords, aa_width=None, aa_widths=None, **kwargs
    ) -> np.ndarray:
        n = coords.shape[0]
        if aa_width is not None and aa_widths is not None:
            raise RuntimeError(f"Only one of aa_width {aa_width} and {aa_widths}")
        aa_widths = to_vector(aa_width, aa_widths, n, default=0.0, dtype=np.float32)
        return aa_widths

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
        assert len(locations) == len(blocks)
        if alpha is not None and alphas is not None:
            raise RuntimeError(f"Only one of alpha {alpha} and {alphas}")
        alphas = to_list(alpha, alphas, len(locations))
        assert len(alphas) == len(locations)
        pixel_buffer = []
        alpha_buffer = []
        ipixel = 0
        coords = np.empty((len(locations), 4), dtype=np.int32)
        ipixels = np.empty(len(locations), dtype=np.int32)
        for i, (block, location, alpha) in enumerate(
            zip(blocks, locations, alphas, strict=False)
        ):
            height, width = block.shape[0], block.shape[1]
            x, y = location
            coords[i, :] = [x, y, width, height]
            ipixels[i] = ipixel
            ipixel += block.size
            pixel_buffer.append(block.ravel())
            if not isinstance(alpha, np.ndarray):
                alpha = arrayify1d(alpha, block.size, default=1.0, dtype=np.float32)
            alpha_buffer.append(alpha.ravel())
        pixel_buffer = np.concatenate(pixel_buffer)
        alpha_buffer = np.concatenate(alpha_buffer)
        buffer = np.column_stack([pixel_buffer, alpha_buffer])
        return coords, buffer, ipixels

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
        if alpha is not None and alphas is not None:
            raise RuntimeError(f"Only one of color {alpha} and {alphas}")
        alphas = to_list(alpha, alphas, len(sprites))
        assert len(alphas) == len(sprites)
        pixel_buffer = []
        alpha_buffer = []
        ipixel = 0
        ipixels = []
        for sprite, alpha in zip(sprites, alphas, strict=False):
            sprite = sprite.ravel()
            ipixels.append(ipixel)
            ipixel += len(sprite)
            pixel_buffer.append(sprite)
            alpha = arrayify1d(alpha, len(sprite), default=1.0)
            alpha_buffer.append(alpha.ravel())
        pixel_buffer = np.concatenate(pixel_buffer)
        alpha_buffer = np.concatenate(alpha_buffer)
        buffer = np.column_stack([pixel_buffer, alpha_buffer])

        coords = []
        for isprite, x, y in locations:
            sprite = sprites[isprite]
            height, width = sprite.shape
            coords.append((x, y, width, height))
            ipixels.append(ipixels[isprite])
        coords = self.coerce_coords(coords, 4, np.int32)
        ipixels = np.array(ipixels, dtype=np.int32)
        return coords, buffer, ipixels

    def setup_texts(
        self,
        locations,
        texts,
        font=None,
        fonts=None,
        size=None,
        sizes=None,
        alpha=None,
        alphas=None,
        color=None,
        colors=None,
        rotation=None,
        rotations=None,
        **kwargs,
    ):
        n = len(locations)
        fonts = to_list(font, fonts, n, default="default")
        sizes = to_list(size, sizes, n, default=32)
        alphas = to_list(alpha, alphas, n, default=None)
        rotations = to_list(rotation, rotations, n, default=0)
        rotations = [compute_360_to_180(rotation) for rotation in rotations]

        bbox_blocks = [
            render_text(text, font_name=font_name, size=size, rotation=rotation)
            for text, font_name, size, rotation in zip(
                texts, fonts, sizes, rotations, strict=False
            )
        ]
        bboxes, blocks = zip(*bbox_blocks, strict=False)
        dxdy = [
            compute_offset(angle, h, w)
            for ((_, _, h, w), angle) in zip(bboxes, rotations, strict=False)
        ]
        locations = [
            (x + dx, y + dy)
            for ((x, y), (dx, dy)) in zip(locations, dxdy, strict=False)
        ]

        if isinstance(alphas, (list, tuple)):
            new_alphas = []
            for alpha_, block in zip(alphas, blocks, strict=False):
                if alpha_ == "mask":
                    alpha_ = block
                elif alpha_ == "bug":
                    alpha_ = np.maximum(block, 0.33)
                new_alphas.append(alpha_)
            alphas = new_alphas
        return locations, blocks, alphas


class Graphics1(AbstractGraphics, Graphics1Setup):
    CHANNELS = 1
    INTERFACE = "graphics"
