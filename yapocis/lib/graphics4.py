"""
Low level wrappers around the OpenCL graphics kernels.
"""
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import numpy as np

import yapocis.rpc.programs
from yapocis.lib.text import render_text
from yapocis.lib.graphics import to_list, to_vector

lines_cl = rects_cl = circles_cl = triangles_cl = ellipses_cl = composite_cl = None
program, _ = yapocis.rpc.programs.import_program("graphics4", lines="lines_cl", rects="rects_cl", circles="circles_cl",
                                              triangles="triangles_cl", ellipses="ellipses_cl", composite="composite_cl")


def coerce_coords(coords, width, dtype=np.float32):
    if not hasattr(coords, "dtype"):
        coords = np.array(coords, dtype=dtype)
    assert len(coords.shape) == 2 and coords.shape[1] == width
    return coords


def arrayify(scalar_or_vector, size):
    scalar_or_vector = 1.0 if scalar_or_vector is None else scalar_or_vector
    if isinstance(scalar_or_vector, np.ndarray):
        if len(scalar_or_vector.shape) == 2:
            vector = scalar_or_vector
        if len(scalar_or_vector) == 4:
            vector = np.zeros((size), dtype=np.float32)
            vector[:,:] = scalar_or_vector
    else:
        vector = np.zeros((size), dtype=np.float32)
        vector[:,:] = scalar_or_vector
    return vector


def coerce_colors(coords, colors):
    colors = arrayify(colors, (coords.shape[0],4))
    return colors


def coerce_colors_alpha(coords, colors, alpha=None):
    ca = arrayify(colors, (coords.shape[0],4))
    return ca


def composite_setup(locations, blocks):
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
        block = block.flatten()
        pixel_buffer.append(block)
        x, y = location
        coords.append((x, y, width, height))
        ipixels.append(ipixel)
        ipixel += len(block)//4
    buffer = np.concatenate(pixel_buffer)
    coords = coerce_coords(coords, 4, np.int32)
    ipixels = np.array(ipixels, dtype=np.int32)
    return coords, buffer, ipixels


def sprites_setup(locations, sprites):
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
        sprite = sprite.flatten()
        ipixels.append(ipixel)
        ipixel += len(sprite)//4
        pixel_buffer.append(sprite.flatten())
    buffer = np.concatenate(pixel_buffer)
    coords = []
    for isprite, x, y in locations:
        sprite = sprites[isprite]
        height, width, depth = sprite.shape
        assert depth==4
        coords.append((x, y, width, height))
        ipixels.append(ipixels[isprite])
    coords = coerce_coords(coords, 4, np.int32)
    ipixels = np.array(ipixels, dtype=np.int32)
    return coords, buffer, ipixels


def lines(image, coords, colors, width=1.0):
    if not coords:
        return
    coords = coerce_coords(coords, 4)
    colors = coerce_colors_alpha(coords, colors)
    lines_cl(image, coords, width, colors, global_size=image.size//4)


def rects(image, coords, colors):
    if not coords:
        return
    coords = coerce_coords(coords, 4)
    colors = coerce_colors_alpha(coords, colors)
    rects_cl(image, coords, colors, global_size=image.size//4)


def circles(image, coords, colors):
    if not coords:
        return
    coords = coerce_coords(coords, 3)
    colors = coerce_colors_alpha(coords, colors)
    circles_cl(image, coords, colors, global_size=image.size//4)


def triangles(image, coords, colors):
    if not coords:
        return
    coords = coerce_coords(coords, 6)
    colors = coerce_colors_alpha(coords, colors)
    triangles_cl(image, coords, colors, global_size=image.size//4)


def ellipses(image, coords, colors):
    coords = coerce_coords(coords, 4)
    colors = coerce_colors_alpha(coords, colors)
    ellipses_cl(image, coords, colors, global_size=image.size//4)


def composite(image, locations, blocks):
    """
    Composite blocks of pixels into target image

    Args:
        image: 2d array of pixels
        coords: a tuple composed of a location (x,y)
        blocks: list of pixel blocks
        alpha: list of scalar or vector of opacity

    Returns: None

    """
    coords, buffer, ipixels = composite_setup(locations, blocks)
    composite_cl(image, coords, buffer, ipixels, global_size=image.size//4)


def sprites(image, locations, sprites):
    """
    Composite sprites into target image

    Args:
        image: 2d array of pixels
        coords: a tuple composed of a location (sprite_index, x,y)
        blocks: list of sprites blocks
    Returns: None

    """
    coords, buffer, ipixels = sprites_setup(locations, sprites)
    composite_cl(image, coords, buffer, ipixels, global_size=image.size//4)


def setup_texts(locations, texts, font=None, fonts=None, size=None, sizes=None,
                color=None, colors=None, alpha=None, alphas=None, rotation=None, rotations=None):
    n = len(locations)
    n = len(locations)
    colors = to_list(color, colors, n, default=(1,1,1,1))
    fonts = to_list(font,fonts,n,default="default")
    sizes = to_list(size,sizes,n, default=32)
    alphas = to_list(alpha, alphas, n, default="mask")
    rotations = to_list(rotation, rotations, n, default=0.0)
    colors = np.asarray(colors, dtype=np.float32)
    assert colors.shape == (len(locations),4)
    blocks = [render_text(text, font_name, size,rotation=rotation) for text, font_name, size, rotation in zip(texts, fonts, sizes, rotations)]
    bboxes,blocks = zip(*blocks)
    new_locations = []
    for ilocation in range(len(locations)):
        location = locations[ilocation]
        rotation = rotations[ilocation]
        if rotation is None:
            new_locations.append(location)
        else:
            bbox = bboxes[ilocation]
            block = blocks[ilocation]
            h,w = block.shape
            ww,ll = bbox[2:]
            radians = 2 * np.pi * (rotation/360)
            dy = np.sin(radians) * ww/2
            dx = np.cos(radians) * ww/2
            x,y = location
            x -= w//2
            y -= h//2
            x += int(dx)
            y -= int(dy)
            new_locations.append((x,y))
    locations = new_locations

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
    return locations, new_blocks

def draw_strings(image, locations, texts, **kwargs):
    locations, blocks = setup_texts(locations, texts, **kwargs)
    composite(image, locations, blocks)


