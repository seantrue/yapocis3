"""
Low level wrappers around the OpenCL graphics kernels.
"""
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import numpy as np

import yapocis.rpc.programs
from yapocis.lib.text import render_text

lines_cl = rects_cl = circles_cl = triangles_cl = ellipses_cl = composite_cl = None
program, _ = yapocis.rpc.programs.import_program("graphics",
                                                 lines="lines_cl", rects="rects_cl", circles="circles_cl",
                                                 triangles="triangles_cl", ellipses="ellipses_cl",
                                                 composite="composite_cl")


def coerce_coords(coords, width, dtype=np.int32):
    if not hasattr(coords, "dtype"):
        coords = np.array(coords, dtype=dtype)
    assert len(coords.shape) == 2 and coords.shape[1] == width
    return coords


def arrayify1d(scalar_or_vector, length, default=None, dtype=np.float32):
    scalar_or_vector = default if scalar_or_vector is None else scalar_or_vector
    if scalar_or_vector is None:
        raise ValueError(f"Both value and default are None")
    if isinstance(scalar_or_vector, np.ndarray):
        vector = scalar_or_vector.astype(dtype)
    elif isinstance(scalar_or_vector, (list, tuple)):
        vector = np.array(scalar_or_vector, dtype=dtype)
    else:
        vector = [scalar_or_vector] * length
        vector = np.asarray(vector)
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
    return coerced


def to_vector(scalar, vector, n, dtype=np.float32, default=None):
    if scalar is None and vector is None:
        scalar = default
    coerced = arrayify1d(scalar, n, dtype=dtype) if scalar else arrayify1d(vector, n, dtype=dtype)
    return coerced


def coerce_colors(coords, color=None, colors=None, alpha=None, alphas=None, **kwargs) -> np.ndarray:
    n = coords.shape[0]
    if color and colors:
        raise RuntimeError(f"Only one of color {color} and {colors}")
    if alpha and alphas:
        raise RuntimeError(f"Only one of color {alpha} and {alphas}")
    colors = to_vector(color, colors, n)
    alphas = to_vector(alpha, alphas, n, default=1.0)
    ca = np.column_stack([colors, alphas])
    return ca


def composite_setup(locations, blocks, alpha=None, alphas=None):
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
    assert len(locations) == len(blocks)
    if alpha is not None and alphas is not None:
        raise RuntimeError(f"Only one of alpha {alpha} and {alphas}")
    alphas = to_list(alpha, alphas, len(locations))
    assert len(alphas) == len(locations)
    pixel_buffer = []
    alpha_buffer = []
    ipixel = 0
    coords = []
    ipixels = []
    for block, location, alpha in zip(blocks, locations, alphas):
        height, width = block.shape
        x, y = location
        coords.append((x, y, width, height))
        ipixels.append(ipixel)
        ipixel += block.size
        pixel_buffer.append(block.flatten())
        if not isinstance(alpha, np.ndarray):
            alpha = arrayify1d(alpha, block.size, default=1.0, dtype=None)
        alpha_buffer.append(alpha.flatten())
    pixel_buffer = np.concatenate(pixel_buffer)
    alpha_buffer = np.concatenate(alpha_buffer)
    buffer = np.column_stack([pixel_buffer, alpha_buffer])
    coords = coerce_coords(coords, 4, np.int32)
    ipixels = np.array(ipixels, dtype=np.int32)
    return coords, buffer, ipixels


def sprites_setup(locations, sprites, alpha=None, alphas=None):
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
        sprite = sprite.flatten()
        ipixels.append(ipixel)
        ipixel += len(sprite)
        pixel_buffer.append(sprite.flatten())
        alpha = arrayify1d(alpha, len(sprite), default=1.0)
        alpha_buffer.append(alpha.flatten())
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


def lines(image, coords, width=1.0, **kwargs):
    if not coords:
        return
    coords = coerce_coords(coords, 4)
    colors = coerce_colors(coords, **kwargs)
    lines_cl(image, coords, width, colors)


def rects(image, coords, **kwargs):
    if not coords:
        return
    coords = coerce_coords(coords, 4)
    colors = coerce_colors(coords, **kwargs)
    rects_cl(image, coords, colors)


def circles(image, coords, **kwargs):
    if not coords:
        return
    coords = coerce_coords(coords, 3)
    colors = coerce_colors(coords, **kwargs)
    circles_cl(image, coords, colors)


def triangles(image, coords, **kwargs):
    if not coords:
        return
    coords = coerce_coords(coords, 6)
    colors = coerce_colors(coords, **kwargs)
    triangles_cl(image, coords, colors)


def ellipses(image, coords, **kwargs):
    coords = coerce_coords(coords, 4)
    colors = coerce_colors(coords, **kwargs)
    ellipses_cl(image, coords, colors)


def composite(image, locations, blocks, **kwargs):
    """
    Composite blocks of pixels into target image

    Args:
        image: 2d array of pixels
        coords: a tuple composed of a location (x,y)
        blocks: list of pixel blocks
        alpha: list of scalar or vector of opacity

    Returns: None

    """
    coords, buffer, ipixels = composite_setup(locations, blocks, **kwargs)
    composite_cl(image, coords, buffer, ipixels)


def sprites(image, locations, sprites, alpha=None, alphas=None):
    """
    Composite sprites into target image

    Args:
        image: 2d array of pixels
        coords: a tuple composed of a location (sprite_index, x,y)
        blocks: list of sprites blocks
        alpha: optional list of scalar or vector of opacity
    Returns: None

    """
    coords, buffer, ipixels = sprites_setup(locations, sprites, alpha=alpha, alphas=alphas)
    composite_cl(image, coords, buffer, ipixels)


def setup_texts(locations, texts, font=None, fonts=None, size=None, sizes=None, alpha=None, alphas=None, color=None,
                colors=None, rotation=None, rotations=None):
    n = len(locations)
    fonts = to_list(font, fonts, n, default="default")
    sizes = to_list(size, sizes, n, default=32)
    alphas = to_list(alpha, alphas, n, default=None)
    rotations = to_list(rotation, rotations, n, default=None)
    bbox_blocks = [render_text(text, font_name=font_name, size=size, rotation=rotation)
                   for text, font_name, size, rotation in zip(texts, fonts, sizes, rotations)]
    bboxes, blocks = zip(*bbox_blocks)
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


def draw_strings(image, locations, texts, **kwargs):
    locations, blocks, alphas = setup_texts(locations, texts, **kwargs)
    composite(image, locations, blocks, alphas=alphas)
