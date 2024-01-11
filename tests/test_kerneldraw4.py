#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

import time

import numpy as np

from .test_graphics4 import (random_alpha, random_circles, random_colors,
                             random_composites, random_ellipses,
                             random_lines, random_rects, random_sprites,
                             random_triangles)
from yapocis.lib.kerneldraw4 import KernelDraw4
from yapocis.utils.stereo_typing import Optional, Image4

from . import ASIZE, StopWatcher, snapshot, snapshow


def just_canvas() -> Image4:
    return np.zeros((ASIZE, ASIZE, 4), dtype=np.float32)


def setup_test(device_name: Optional[str] = None, chunk_size: Optional[int] = None) -> tuple[Image4, KernelDraw4]:
    canvas = just_canvas()
    return canvas, KernelDraw4(canvas, device_name=device_name, chunk_size=chunk_size)


def test_batch():
    canvas, ad = setup_test()
    random_coords = random_ellipses()
    circles = random_circles()
    rects = random_rects()
    lines = random_lines()
    locations, blocks = random_composites()
    sprite_locations, sprites = random_sprites()
    with StopWatcher() as show:
        ad.composite(locations, blocks)
        ad.sprites(sprite_locations, sprites)
        ad.ellipses(random_coords, colors=random_colors())
        ad.circles(circles, colors=random_colors())
        ad.rects(rects, colors=random_colors())
        ad.lines(lines, colors=random_colors(), distance=1.5)
        ad.render()
        a = ad.read()
        show(ad, a)


def test_displaylist():
    canvas, ad = setup_test()
    rects = random_rects()
    lines = random_lines()
    with StopWatcher() as show:
        for xyxy, color in zip(lines, random_colors()):
            ad.line(xyxy, color=color, distance=1.5)
        for rect, color in zip(rects, random_colors()):
            ad.rect(rect, color=color)
        for x in range(0, ASIZE, 100):
            ad.draw_string((x, x), "Hello world!", color=[1, 1, 1, .5])
        ad.render()
        a = ad.read()
        show(ad, a)


def test_chunking():
    ELEMENTS = 1000
    rects = random_rects(ELEMENTS)
    lines = random_lines(ELEMENTS)
    colors = random_colors(ELEMENTS)
    alphas = random_alpha(ELEMENTS)

    times = []
    for chunk_size in (None, 10, 100, 500):
        canvas = just_canvas()
        t = time.time()
        ad = KernelDraw4(canvas, chunk_size=chunk_size)
        with StopWatcher() as show:
            for xyxy, color, alpha in zip(lines, colors, alphas):
                ad.line(xyxy, distance=1.5, color=color, alpha=alpha)
            for rect, color, alpha in zip(rects, colors, alphas):
                ad.rect(rect, color=color, alpha=alpha)
            for x in range(0, ASIZE, 100):
                ad.draw_string((x, x), "Hello world!")
            ad.render()
            canvas = ad.read()
            times.append(time.time() - t)
            show(ad, f"chunked {chunk_size}", canvas)
    print(times)
