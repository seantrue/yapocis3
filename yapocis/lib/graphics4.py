"""
Low level wrappers around the OpenCL graphics kernels.
"""
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

import numpy as np

from yapocis.lib.abstract_graphics import (AbstractGraphics,
                                           AbstractGraphicsSetup, to_list,
                                           to_vector)
from yapocis.lib.typeset import compute_offset, render_text
from yapocis.utils.basic import compute_360_to_180


def arrayify(scalar_or_vector, size, dtype=np.float32):
    scalar_or_vector = dtype(1.0) if scalar_or_vector is None else scalar_or_vector
    if isinstance(scalar_or_vector, np.ndarray):
        if len(scalar_or_vector.shape) == 2:
            vector = scalar_or_vector
        elif len(scalar_or_vector) == 4:
            vector = np.empty(size, dtype=dtype)
            vector[:,:] = scalar_or_vector
    else:
        vector = np.empty(size, dtype=dtype)
        vector[:,:] = scalar_or_vector
    return vector



class GraphicsSetup4(AbstractGraphicsSetup):

    def coerce_coords(self, coords, width, dtype=np.float32, **kwargs):
        if not hasattr(coords, "dtype"):
            coords = np.array(coords, dtype=dtype)
        assert len(coords.shape) == 2 and coords.shape[1] == width
        return coords

    def coerce_colors(self, coords, color=None, colors=None, **kwargs):
        n = coords.shape[0]
        if color and colors:
            raise RuntimeError(f"Only one of color {color} and {colors}")
        if color is None and colors is None:
            raise RuntimeError(f"Missing one of color or colors")
        if color:
            color = to_vector(color, color, self.CHANNELS, default=1.0, dtype=np.float32)
            colors = np.empty((n,self.CHANNELS), dtype=np.float32)
            colors[:] = color
        colors = arrayify(colors, (coords.shape[0],4))
        return colors

    def coerce_widths(self, coords, width=None, widths=None, distance=None, **kwargs) -> np.ndarray:
        n = coords.shape[0]
        if width and widths:
            raise RuntimeError(f"Only one of width {width} and {widths}")
        if width and distance:
            raise RuntimeError(f"Only one of width/distance {width} and {distance}")
        width = distance if width is None else width
        widths = to_vector(width, widths, n, default=1.0, dtype=np.float32)
        return widths


    def setup_composite(self, locations, blocks, **kwargs):
        """
        Compute the information for compositing a set of distinct blocks

        Args:
            coords: list of tuples composed of a location (x,y)
            blocks: list of pixel blocks, one per element of coords

        Returns:
            coords: list of tuples (x,y,block_width,block_height)
            buffer: float32 array of intensity and opacity per point (2xN)
            ipixels: index of start of block in buffer

        """
        assert len(locations) == len(blocks)
        pixel_buffer = []
        ipixel = 0
        coords = []
        ipixels = []
        for block, location in zip(blocks, locations):
            height,width,depth = block.shape
            assert depth == 4
            block = block.ravel()
            pixel_buffer.append(block)
            x, y = location
            coords.append((x, y, width, height))
            ipixels.append(ipixel)
            ipixel += len(block)//4
        buffer = np.concatenate(pixel_buffer)
        coords = self.coerce_coords(coords, 4, np.int32)
        ipixels = np.array(ipixels, dtype=np.int32)
        return coords, buffer, ipixels


    def setup_sprites(self, locations, sprites, **kwargs):
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

        pixel_buffer = []
        ipixel = 0
        ipixels = []
        for sprite in sprites:
            sprite = sprite.ravel()
            ipixels.append(ipixel)
            ipixel += len(sprite)//4
            pixel_buffer.append(sprite)
        buffer = np.concatenate(pixel_buffer)
        coords = []
        for isprite, x, y in locations:
            sprite = sprites[isprite]
            height, width, depth = sprite.shape
            assert depth==4
            coords.append((x, y, width, height))
            ipixels.append(ipixels[isprite])
        coords = self.coerce_coords(coords, 4, np.int32)
        ipixels = np.array(ipixels, dtype=np.int32)
        return coords, buffer, ipixels

    def setup_texts(lself, locations, texts, font=None, fonts=None, size=None, sizes=None,
                    color=None, colors=None, alpha=None, alphas=None, rotation=None, rotations=None, **kwargs):
        n = len(locations)
        n = len(locations)
        colors = to_list(color, colors, n, default=(1,1,1,1))
        fonts = to_list(font,fonts,n,default="default")
        sizes = to_list(size,sizes,n, default=32)
        alphas = to_list(alpha, alphas, n, default="mask")
        rotations = to_list(rotation, rotations, n, default=0.0)
        colors = np.asarray(colors, dtype=np.float32)
        assert colors.shape == (len(locations),4)
        rotations = to_list(rotation, rotations, n, default=0)
        rotations = [compute_360_to_180(rotation) for rotation in rotations]

        bbox_blocks = [render_text(text, font_name=font_name, size=size, rotation=rotation)
                       for text, font_name, size, rotation in zip(texts, fonts, sizes, rotations)]
        bboxes, blocks = zip(*bbox_blocks)
        dxdy = [compute_offset(angle, h, w) for ((_, _, h, w), angle) in zip(bboxes, rotations)]
        locations = [(x + dx, y + dy) for ((x, y), (dx, dy)) in zip(locations, dxdy)]


        if isinstance(alphas, (list, tuple)):
            new_alphas = []
            assert len(alphas) == len(blocks) == colors.shape[0]
            for itext in range(len(blocks)):
                alpha  = alphas[itext]
                block = blocks[itext]
                color = colors[itext,:]
                if alpha == "mask":
                    alpha = block
                elif alpha == "bug":
                    alpha = np.maximum(block,.33)
                elif isinstance(alpha, float):
                    new_alpha = np.empty_like(block)
                    new_alpha[:,:] = alpha
                    alpha = new_alpha
                elif alpha is None:
                    new_alpha = np.empty_like(block)
                    new_alpha[:,:] = color[3]
                    alpha = new_alpha
                new_alphas.append(alpha)
            alphas = new_alphas
        new_blocks = []
        for itext in range(len(blocks)):
            alpha = alphas[itext]
            block = blocks[itext]
            color = colors[itext, :]
            shape = list(block.shape)
            shape.append(4)
            rgba = np.empty(tuple(shape), dtype=np.float32)
            rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2] = block,block,block
            rgba[:,:,:3] *= color[:3]
            rgba[:,:,3] = alpha
            new_blocks.append(rgba)
        return locations, new_blocks, None

class Graphics4(AbstractGraphics, GraphicsSetup4):
    INTERFACE="graphics4"
    CHANNELS=4
