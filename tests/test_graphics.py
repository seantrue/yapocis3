#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import pytest

import numpy as np
from random import randint, shuffle

from yapocis.utils.basic import show_array, imread
from yapocis.lib.channels import split_channels
from yapocis.lib.hsi import rgb2hsi
from yapocis.lib.graphics import lines, rects, circles, triangles, ellipses, composite, sprites, draw_strings
from . import image_test


ASIZE = 1000


def setup():
    return np.zeros((ASIZE, ASIZE)).astype(np.float32)


def image_source(align=5):
    image = image_test()
    r, g, b = split_channels(image)
    h, s, i = rgb2hsi(r, g, b)
    d = min(*i.shape)
    d = d - int(d % align);
    i = i[:d, :d]
    return i


def image_sprites(steps=4):
    image = image_source(align=steps)
    dimension = image.shape[0]
    step = dimension // steps
    blocks = []
    for x in range(0, dimension, step):
        for y in range(0, dimension, step):
            blocks.append(image[x:x + step, y:y + step])
    return blocks


def random_points(n=100):
    def rx():
        return randint(0, ASIZE)

    def ry():
        return randint(0, ASIZE)

    points = []
    for iline in range(n):
        x, y = rx(), ry()
        points.append([x, y])
    return points


def random_colors(n=100):
    return np.random.uniform(0, 1.0, n).astype(np.float32)


def random_alpha(n=100):
    return np.random.uniform(.25, 1.0, n).astype(np.float32)


def random_lines(n=100):
    lines = random_points(n)
    length = 50
    for line in lines:
        x, y = line
        x1, y1 = x + randint(-length, length), y + randint(-length, length)
        line += [x1, y1]
    return lines


random_rects = random_lines


def random_triangles(n=100):
    lines = random_points(n)
    a = np.zeros((ASIZE, ASIZE)).astype(np.float32)
    length = 50
    for line in lines:
        x, y = line
        x1, y1 = x + randint(-length, length), y + randint(-length, length)
        line += [x1, y1]
        x1, y1 = x + randint(-length, length), y + randint(-length, length)
        line += [x1, y1]
    return lines


def random_circles(n=100):
    circles = random_points(n)
    radius = 50
    for circle in circles:
        circle.append(randint(2, radius))
    return circles


def random_ellipses(n=100):
    ellipses = random_points(n=n)
    radius = 50
    for ellipse in ellipses:
        ellipse += [randint(2, radius), randint(2, radius)]
    return ellipses


def random_composites(n=100):
    locations = random_points(n)
    colors = random_colors(n)
    sprites = image_sprites()
    blocks = sprites[:]
    while len(blocks) < n:
        blocks += sprites
    blocks = blocks[:n]
    return locations, blocks


def random_sprites(n=100):
    random_coords, random_sprites = random_composites(n)
    random_sprites = random_sprites[:10]
    for i in range(len(random_coords)):
        j = int(i % len(random_sprites))
        random_coords[i].insert(0, j)
    return random_coords, random_sprites


def test_lines():
    canvas = setup()
    random_coords = random_lines()
    coords = []
    for i in range(100, ASIZE, 100):
        for j in range(100, ASIZE, 100):
            coords.append([0, 0, i, j])
    lines(canvas, coords, 1.5, color=1.0, alpha=1.0)
    lines(canvas, random_coords, 1.5, color=1.0, alphas=random_alpha())
    show_array(f"draw", canvas)


def test_rects():
    canvas = setup()
    random_coords = random_lines()
    coords = []
    for i in range(100, ASIZE, 100):
        for j in range(100, ASIZE, 100):
            coords.append([i - 25, j - 25, i + 25, j + 25])
    rects(canvas, coords, color=1.0, alpha=0.5)
    rects(canvas, random_coords, colors=random_colors(), alphas=random_alpha())
    show_array(f"rects", canvas)


def test_circles():
    canvas = setup()
    random_coords = random_circles()
    coords = []
    for i in range(100, ASIZE, 100):
        for j in range(100, ASIZE, 100):
            coords.append([i, j, 100])
    circles(canvas, coords, color=1.0, alpha=0.5)
    circles(canvas, random_coords, colors=random_colors(), alphas=random_alpha())
    show_array(f"circles", canvas)


def test_triangles():
    canvas = setup()
    random_coords = random_triangles()
    coords = []
    for i in range(100, ASIZE, 100):
        for j in range(100, ASIZE, 100):
            coords.append([i, j, i + 50, j + 0, i + 50, j - 50])
    triangles(canvas, coords, color=1.0, alpha=0.5)
    triangles(canvas, random_coords, colors=random_colors(), alphas=random_alpha())
    show_array(f"triangles", canvas)


def test_ellipses():
    canvas = setup()
    random_coords = random_ellipses()
    coords = []
    for i in range(100, ASIZE, 100):
        for j in range(100, ASIZE, 100):
            coords.append([i, j, 50, 100])
    ellipses(canvas, coords, color=1.0, alpha=0.5)
    coords = []
    for i in range(100, ASIZE, 100):
        coords.append([i, i, 100, 50])
    colors = np.linspace(0.0, 1.0, len(coords))
    alpha = 1.0
    ellipses(canvas, coords, colors=colors, alpha=alpha)
    ellipses(canvas, random_coords, colors=random_colors(), alphas=random_alpha())
    show_array(f"ellipses", canvas)


def test_composite():
    canvas = setup()
    random_coords, random_blocks = random_composites()
    composite(canvas, random_coords, random_blocks)
    show_array("composite", canvas)


def test_composite_sprites():
    canvas = setup()
    random_coords, random_sprites_ = random_sprites()
    sprites(canvas, random_coords, random_sprites_)
    show_array("sprites", canvas)


def test_composite_mosaic():
    canvas = setup()
    image = image_test()
    from random import shuffle
    r, g, b = split_channels(image)
    h, s, i = rgb2hsi(r, g, b)
    d = min(*i.shape)
    d = d - int(d % 5);
    i = i[:d, :d]
    show_array("i", i)
    iout = np.zeros_like(i)
    step = d // 5
    blocks = [i]
    coords = [(0, 0)]
    for x in range(0, d, step):
        for y in range(0, d, step):
            block = i[x:x + step, y:y + step]
            coords.append((x, y))
            blocks.append(block)
    shuffle(blocks)
    composite(iout, coords, blocks)
    show_array("mosaic", iout)


def test_simple_texts():
    canvas = setup()
    locations = []
    s = "Hello world"
    for x in range(100, ASIZE, 100):
        locations.append((x // 2, x))
    strings = [s] * len(locations)
    draw_strings(canvas, locations, strings, font="default", size=32, alpha="bug")
    show_array("simple texts", canvas)


def test_texts():
    canvas = setup()
    locations = random_points(300)
    strings = ["Hello", "sympathy", "goodbye", "Aqua"]
    fonts = ["bold", "italic", "light", "regular"]
    sizes = [18, 36, 72, 112]
    alphas = ["bug", "mask", .5, 1.0]
    while len(strings) < len(locations):
        strings += strings
        sizes += sizes
        fonts += fonts
        alphas += alphas
    strings = strings[:len(locations)]
    sizes = sizes[:len(strings)]
    fonts = fonts[:len(sizes)]
    alphas = alphas[:len(sizes)]
    shuffle(strings)
    shuffle(sizes)
    shuffle(fonts)
    shuffle(alphas)
    draw_strings(canvas, locations, strings, fonts=fonts, sizes=sizes, alphas=alphas)
    show_array("texts", canvas)


