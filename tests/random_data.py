#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.
"""Centralized random data generation for tests with reproducible seeding."""

import os
from random import randint
from random import seed as random_seed
from random import shuffle

import numpy as np

# Import test dependencies
from . import (
    ASIZE,
    double_shuffle,
    image_test,
    interesting_words,
)


def validating(func):
    def wrapper(*args, **kwargs):
        if os.getenv("YAPOCIS_VALIDATION"):
            random_seed(42)
            np.random.seed(42)
        result = func(*args, **kwargs)
        return result

    return wrapper


# set reproducible seed if in validation mode

# ==============================================================================
# Common functions (shared by graphics1 and graphics4)
# ==============================================================================


@validating
def random_points(n=100):
    """Generate n random (x, y) points."""

    def rx():
        return randint(0, ASIZE)

    def ry():
        return randint(0, ASIZE)

    points = []
    for iline in range(n):
        x, y = rx(), ry()
        points.append([x, y])
    return points


@validating
def random_lines(n=100):
    """Generate n random line segments (x0, y0, x1, y1)."""
    lines = random_points(n)
    length = 50
    for line in lines:
        x, y = line
        x1, y1 = x + randint(-length, length), y + randint(-length, length)
        line += [x1, y1]
    return lines


@validating
def random_widths(n=100):
    """Generate n random line widths (1.0 to 10.0)."""
    return np.random.uniform(1.0, 10.0, n).astype(np.float32)


@validating
def random_triangles(n=100):
    """Generate n random triangles (x0, y0, x1, y1, x2, y2)."""
    lines = random_points(n)
    length = 50
    for line in lines:
        x, y = line
        x1, y1 = x + randint(-length, length), y + randint(-length, length)
        line += [x1, y1]
        x1, y1 = x + randint(-length, length), y + randint(-length, length)
        line += [x1, y1]
    return lines


@validating
def random_circles(n=100):
    """Generate n random circles (x, y, radius)."""
    circles = random_points(n)
    radius = 50
    for circle in circles:
        circle.append(randint(2, radius))
    return circles


@validating
def random_ellipses(n=100):
    """Generate n random ellipses (x, y, radius_x, radius_y)."""
    ellipses = random_points(n=n)
    radius = 50
    for ellipse in ellipses:
        ellipse += [randint(2, radius), randint(2, radius)]
    return ellipses


# Aliases
random_rects = random_lines

# ==============================================================================
# Graphics1 (single channel) functions
# ==============================================================================


@validating
def random_colors_1ch(n=100):
    """Generate n random grayscale colors (0.0 to 1.0)."""
    return np.random.uniform(0, 1.0, n).astype(np.float32)


@validating
def random_alpha_1ch(n=100):
    """Generate n random alpha values (0.25 to 1.0)."""
    return np.random.uniform(0.25, 1.0, n).astype(np.float32)


def image_source_1ch(align=5):
    """Create a single-channel test image aligned to grid."""
    from yapocis.lib.channels import split_channels
    from yapocis.lib.hsi import HSI

    hsis = HSI()
    image = image_test()
    r, g, b = split_channels(image)
    h, s, i = hsis.rgb2hsi(r, g, b)
    d = min(*i.shape)
    d = d - int(d % align)
    i = i[:d, :d]
    return i


def image_sprites_1ch(steps=4):
    """Split single-channel test image into sprite blocks."""
    image = image_source_1ch(align=steps)
    dimension = image.shape[0]
    step = dimension // steps
    blocks = []
    for x in range(0, dimension, step):
        for y in range(0, dimension, step):
            blocks.append(image[x : x + step, y : y + step])
    return blocks


@validating
def random_composites_1ch(n=100):
    """Generate random composite locations and sprite blocks (single channel)."""
    locations = random_points(n)
    sprites = image_sprites_1ch()
    blocks = sprites[:]
    while len(blocks) < n:
        blocks += sprites
    blocks = blocks[:n]
    return locations, blocks


@validating
def random_sprites_1ch(n=100):
    """Generate random sprite indices and locations (single channel)."""
    random_coords, random_sprites = random_composites_1ch(n)
    random_sprites = random_sprites[:10]
    for i in range(len(random_coords)):
        j = int(i % len(random_sprites))
        random_coords[i].insert(0, j)
    return random_coords, random_sprites


# ==============================================================================
# Graphics4 (RGBA) functions
# ==============================================================================


@validating
def random_vector(n=100):
    """Generate n random values (0.2 to 1.0) for RGBA components."""
    return np.random.uniform(0.2, 1.0, n).astype(np.float32)


def components_to_colors(*components):
    """Convert separate R, G, B, A component arrays into RGBA color array."""
    colors = list(zip(*components, strict=False))
    colors = np.array(colors)
    return colors


@validating
def random_colors_4ch(n=100):
    """Generate n random RGBA colors."""
    colors = components_to_colors(
        random_vector(n), random_vector(n), random_vector(n), random_vector(n)
    )
    return colors


@validating
def random_alpha_4ch(n=100):
    """Generate n random alpha values (0.2 to 1.0) for RGBA."""
    return random_vector(n)


@validating
def random_icons(n=100):
    """Generate n random icon sprites with shuffled selection."""
    from yapocis.icons import icon, icons

    icon_names = icons()
    shuffle(icon_names)
    icon_names = icon_names[:n]
    colors = random_colors_4ch(n)
    icns = [icon(icon_name, size=100) for icon_name in icon_names]
    icns = double_shuffle(icns, n)
    return icns


def image_source_4ch(align=5):
    """Create a 4-channel RGBA test image aligned to grid."""
    image = image_test()
    d = min(image.shape[0], image.shape[1])
    d = d - int(d % align)
    image4 = np.zeros((d, d, 4), dtype=np.float32)
    image4[:, :, :3] = image[:d, :d, :]
    image4[:, :, 3] = 1.0
    return image4


def image_sprites_4ch(steps=4):
    """Split 4-channel test image into sprite blocks with locations."""
    image = image_source_4ch(align=steps)
    dimension = image.shape[0]
    step = dimension // steps
    sprites = []
    for x in range(0, dimension, step):
        for y in range(0, dimension, step):
            sprites.append(((x, y), image[x : x + step, y : y + step, :]))
    return sprites


@validating
def random_composites_4ch(n=100):
    """Generate random composite locations and sprite blocks (4-channel)."""
    locations = random_points(n)
    sprites = [sprite for _, sprite in image_sprites_4ch(8)]
    blocks = sprites[:]
    blocks = double_shuffle(blocks, n)
    return locations, blocks


@validating
def random_sprites_4ch(n=100):
    """Generate random sprite indices and locations (4-channel)."""
    random_coords, random_sprites = random_composites_4ch(n)
    random_sprites = random_sprites[:n]
    for i in range(len(random_coords)):
        j = int(i % len(random_sprites))
        random_coords[i].insert(0, j)
    return random_coords, random_sprites


# ==============================================================================
# Shared functions (used by both graphics1 and graphics4)
# ==============================================================================


@validating
def random_texts(n=100):
    """Generate random text rendering parameters.

    Returns: (locations, strings, fonts, sizes, rotations, alphas)
    """
    locations = random_points(n)
    strings = interesting_words(n)
    fonts = ["bold", "italic", "light", "regular"]
    sizes = [18, 36, 72, 112]
    alphas = ["bug", "mask", 0.5, 1.0]
    rotations = double_shuffle(list(range(0, 360, 30)), n)
    strings = double_shuffle(strings, n)
    sizes = double_shuffle(sizes, n)
    fonts = double_shuffle(fonts, n)
    alphas = double_shuffle(alphas, n)
    return locations, strings, fonts, sizes, rotations, alphas


# ==============================================================================
# Backward compatibility aliases (default to graphics1/single-channel)
# ==============================================================================

random_colors = random_colors_1ch
random_alpha = random_alpha_1ch
image_source = image_source_1ch
image_sprites = image_sprites_1ch
random_composites = random_composites_1ch
random_sprites = random_sprites_1ch
