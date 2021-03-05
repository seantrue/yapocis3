"""
Low level wrappers around the OpenCL graphics kernels.
"""
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import numpy as np

from yapocis.lib.typeset import render_text, compute_offset
from yapocis.utils.basic import compute_360_to_180
from .abstract_graphics import AbstractGraphicsSetup, AbstractGraphics


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
    coerced = arrayify1d(scalar, n, dtype=dtype) if scalar else arrayify1d(vector, n, dtype=dtype)
    return coerced


class Graphics1Setup(AbstractGraphicsSetup):
    CHANNELS=1
    def coerce_coords(self, coords, width, dtype=np.int32, **kwargs):
        if not hasattr(coords, "dtype"):
            coords = np.array(coords, dtype=dtype)
        assert len(coords.shape) == 2 and coords.shape[1] == width
        return coords


    def coerce_colors(self, coords, color=None, colors=None, alpha=None, alphas=None, **kwargs) -> np.ndarray:
        n = coords.shape[0]
        if color and colors:
            raise RuntimeError(f"Only one of color {color} and {colors}")
        if alpha and alphas:
            raise RuntimeError(f"Only one of color {alpha} and {alphas}")
        colors = to_vector(color, colors, n)
        alphas = to_vector(alpha, alphas, n, default=1.0)
        ca = np.column_stack([colors, alphas])
        return ca


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
        for i, (block, location, alpha) in enumerate(zip(blocks, locations, alphas)):
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
        for sprite, alpha in zip(sprites, alphas):
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

    def setup_texts(self, locations, texts, font=None, fonts=None, size=None, sizes=None, alpha=None, alphas=None, color=None,
                    colors=None, rotation=None, rotations=None, **kwargs):
        n = len(locations)
        fonts = to_list(font, fonts, n, default="default")
        sizes = to_list(size, sizes, n, default=32)
        alphas = to_list(alpha, alphas, n, default=None)
        rotations = to_list(rotation, rotations, n, default=0)
        rotations = [compute_360_to_180(rotation) for rotation in rotations]

        bbox_blocks = [render_text(text, font_name=font_name, size=size, rotation=rotation)
                       for text, font_name, size, rotation in zip(texts, fonts, sizes, rotations)]
        bboxes, blocks = zip(*bbox_blocks)
        dxdy = [compute_offset(angle, h, w) for ((_, _, h, w), angle) in zip(bboxes, rotations)]
        locations = [(x + dx, y + dy) for ((x, y), (dx, dy)) in zip(locations, dxdy)]

        if isinstance(alphas, (list, tuple)):
            new_alphas = []
            for alpha_, block in zip(alphas, blocks):
                if alpha_ == "mask":
                    alpha_ = block
                elif alpha_ == "bug":
                    alpha_ = np.maximum(block, .33)
                new_alphas.append(alpha_)
            alphas = new_alphas
        return locations, blocks, alphas



class Graphics1(AbstractGraphics, Graphics1Setup):
    CHANNELS=1
    INTERFACE="graphics"

