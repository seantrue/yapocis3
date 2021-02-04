"""
Low level wrappers around the OpenCL graphics kernels.
"""
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import numpy as np

import yapocis.rpc.programs
from yapocis.lib.text import render_text

lines_cl = rects_cl = circles_cl = triangles_cl = ellipses_cl = composite_cl = None
program, _ = yapocis.rpc.programs.import_program("graphics", lines="lines_cl", rects="rects_cl", circles="circles_cl",
                                              triangles="triangles_cl", ellipses="ellipses_cl", composite="composite_cl")


def coerce_coords(coords, width, dtype=np.float32):
    if not hasattr(coords, "dtype"):
        coords = np.array(coords, dtype=dtype)
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


def composite_setup(locations, blocks, alpha):
    """
    Compute the information for compositing a set of distinct blocks
    Args:
        coords: list of tuples composed of a location (x,y)
        blocks: list of pixel blocks, one per element of coords
        alpha: list of scalar or vector of opacity

    Returns:
        coords: list of tuples (x,y,block_width,block_height)
        buffer: float32 array of intensity and opacity per point (2xN)
        ipixels: index of start of block in buffer

    """
    alpha = 1.0 if alpha is None else alpha
    alphas = alpha if isinstance(alpha, list) else arrayify(alpha, len(locations))
    assert len(locations) == len(blocks)
    assert len(alphas) == len(blocks)
    pixel_buffer = []
    alpha_buffer = []
    ipixel = 0
    coords = []
    ipixels = []
    for block, location, alpha in zip(blocks, locations, alphas):
        height, width = block.shape
        block = block.flatten()
        x, y = location
        coords.append((x, y, width, height))
        ipixels.append(ipixel)
        ipixel += len(block)
        pixel_buffer.append(block.flatten())
        alpha = 1.0 if alpha is None else alpha
        alpha = arrayify(alpha, len(block))
        alpha_buffer.append(alpha.flatten())
    pixel_buffer = np.concatenate(pixel_buffer)
    alpha_buffer = np.concatenate(alpha_buffer)
    buffer = np.column_stack([pixel_buffer, alpha_buffer])
    coords = coerce_coords(coords, 4, np.int32)
    ipixels = np.array(ipixels, dtype=np.int32)
    return coords, buffer, ipixels


def sprites_setup(locations, sprites, alpha):
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

    alpha = 1.0 if alpha is None else alpha
    alphas = arrayify(alpha, len(sprites))
    assert len(alphas) == len(sprites)
    pixel_buffer = []
    alpha_buffer = []
    ipixel = 0
    ipixels = []
    for sprite, alpha in zip(sprites, alphas):
        sprite = sprite.flatten()
        ipixels.append(ipixel)
        ipixel += len(sprite)
        pixel_buffer.append(sprite.flatten())
        alpha = 1.0 if alpha is None else alpha
        alpha = arrayify(alpha, len(sprite))
        alpha_buffer.append(alpha)
    pixel_buffer = np.concatenate(pixel_buffer)
    alpha_buffer = np.concatenate(alpha_buffer)
    buffer = np.column_stack([pixel_buffer, alpha_buffer])

    coords = []
    for isprite, x, y in locations:
        sprite = sprites[isprite]
        height, width = sprite.shape
        coords.append((x, y, width, height))
        ipixels.append(ipixels[isprite])
    coords = coerce_coords(coords, 4, np.int32)
    ipixels = np.array(ipixels, dtype=np.int32)
    return coords, buffer, ipixels


def lines(image, coords, width=1.0, colors=None, alpha=None):
    if not coords:
        return
    coords = coerce_coords(coords, 4)
    colors = coerce_colors_alpha(coords, colors, alpha)
    lines_cl(image, coords, width, colors)


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


def composite(image, locations, blocks, alpha=None):
    """
    Composite blocks of pixels into target image
    Args:
        image: 2d array of pixels
        coords: a tuple composed of a location (x,y)
        blocks: list of pixel blocks
        alpha: list of scalar or vector of opacity

    Returns: None

    """
    coords, buffer, ipixels = composite_setup(locations, blocks, alpha)
    composite_cl(image, coords, buffer, ipixels)


def sprites(image, locations, sprites, alpha=None):
    """
    Composite sprites into target image
    Args:
        image: 2d array of pixels
        coords: a tuple composed of a location (sprite_index, x,y)
        blocks: list of sprites blocks
        alpha: optional list of scalar or vector of opacity
    Returns: None

    """
    coords, buffer, ipixels = sprites_setup(locations, sprites, alpha)
    composite_cl(image, coords, buffer, ipixels)


def setup_texts(locations, texts, fonts, sizes, alpha):
    sizes = 32 if sizes is None else sizes
    alpha = 1.0 if alpha is None else alpha
    if not isinstance(fonts, (list, tuple)):
        fonts = [fonts] * len(locations)
    if not isinstance(sizes, (list, tuple)):
        sizes = [sizes] * len(locations)
    if isinstance(alpha, str):
        alpha = [alpha] * len(locations)
    blocks = [render_text(text, font_name, size) for text, font_name, size in zip(texts, fonts, sizes)]
    if isinstance(alpha, (list, tuple)):
        new_alphas = []
        for alpha_, block in zip(alpha, blocks):
            if alpha_ == "mask":
                alpha_ = block
            elif alpha_ == "bug":
                alpha_ = np.maximum(block,.33)
            new_alphas.append(alpha_)
        alpha = new_alphas
    return locations, blocks, alpha

def draw_strings(image, locations, texts, fonts="default", sizes=32, alpha=1.0):
    locations, blocks, alpha = setup_texts(locations, texts, fonts, sizes, alpha)
    composite(image, locations, blocks, alpha)


