#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.
import time
import numpy as np
import pytest

from tests.test_graphics import random_ellipses, random_circles, random_rects, random_lines, random_composites, \
    random_sprites, random_colors, random_alpha, ASIZE
from yapocis.lib.kerneldraw import KernelDraw
from yapocis.utils.basic import show_array

def setup():
    canvas = np.zeros((ASIZE, ASIZE)).astype(np.float32)
    return canvas, KernelDraw(canvas)


def test_batch():
    random_coords = random_ellipses()
    _, ad = setup()
    circles = random_circles()
    rects = random_rects()
    lines = random_lines()
    locations, blocks = random_composites()
    sprite_locations, sprites = random_sprites()
    ad.ellipses(random_coords, colors=random_colors(), alphas=random_alpha())
    ad.circles(circles, colors=random_colors(), alphas=random_alpha())
    ad.rects(rects, colors=random_colors(), alphas=random_alpha())
    ad.lines(lines, distance=1.5, colors=random_colors(), alphas=random_alpha())
    ad.composite(locations, blocks, alphas=random_alpha())
    ad.sprites(sprite_locations, sprites, alpha=1.0)
    ad.render()
    a = ad.read()
    show_array(f"batch", a)


def test_displaylist():
    random_coords = random_ellipses()
    _, ad = setup()
    circles = random_circles()
    rects = random_rects()
    lines = random_lines()
    for xyxy, color, alpha in zip(lines, random_colors(), random_alpha()):
        ad.line(xyxy, distance=1.5, color=color, alpha=alpha)
    for rect, color, alpha in zip(rects, random_colors(), random_alpha()):
        ad.rect(rect, color=color, alpha=alpha)
    for x in range(100, ASIZE, 100):
        ad.draw_string((x, x), "Hello world!")
    ad.render()
    canvas = ad.read()
    show_array(f"displaylist", canvas)


def test_chunking():
    canvas, _ = setup()
    ellipses = random_ellipses()
    rects = random_rects(1000)
    lines = random_lines(1000)
    circles = random_circles(1000)
    colors = random_colors(1000)
    alphas = random_alpha(1000)
    times = []
    for chunk_size in (None, 10, 100, 500, 10000):
        canvas[:, :] = 0.0
        t = time.time()
        ad = KernelDraw(canvas, chunk_size=chunk_size)
        for xyxy, color, alpha in zip(lines, colors, alphas):
            ad.line(xyxy, distance=1.5, color=color, alpha=alpha)
        for rect, color, alpha in zip(rects, colors, alphas):
            ad.rect(rect, color=color, alpha=alpha)
        for x in range(0, ASIZE, 100):
            ad.draw_string((x, x), "Hello world!")
        ad.render()
        canvas = ad.read()
        times.append(time.time() - t)
    show_array("Chunked", canvas)
    print(times)