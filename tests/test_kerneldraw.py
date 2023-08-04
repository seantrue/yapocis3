#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.
import time

import numpy as np

from .test_graphics import (random_alpha, random_circles, random_colors,
                            random_composites, random_ellipses,
                            random_lines, random_rects, random_sprites,
                            random_texts)
from yapocis.lib.kerneldraw1 import KernelDraw1
from yapocis.utils.stereo_typing import Optional, Image1

from . import ASIZE, StopWatcher


def just_canvas() -> Image1:
    canvas = np.zeros((ASIZE, ASIZE), dtype=np.float32)
    return canvas


def setup(device_name: Optional[str] = None, chunk_size: Optional[int] = None):
    canvas = just_canvas()
    return canvas, KernelDraw1(canvas, device_name=device_name, chunk_size=chunk_size)


def test_batch():
    canvas, ad = setup()
    locations, strings, fonts, sizes, _, text_alphas = random_texts()
    with StopWatcher() as show:
        ad.composite(*random_composites(), alpha=1.0)
        ad.sprites(*random_sprites(), alpha=1.0)
        ad.ellipses(random_ellipses(), colors=random_colors(), alphas=random_alpha())
        ad.circles(random_circles(), colors=random_colors(), alphas=random_alpha())
        ad.rects(random_rects(), colors=random_colors(), alphas=random_alpha())
        ad.lines(random_lines(), colors=random_colors(), alphas=random_alpha())
        ad.draw_strings(locations, strings, fonts=fonts, sizes=sizes, alphas=text_alphas)
        ad.render()
        a = ad.read()
        show(ad, a)


def fill_display_list(ad: KernelDraw1, n: int = 1000) -> None:
    ellipses = random_ellipses(n)
    rects = random_rects(n)
    lines = random_lines(n)
    circles = random_circles(n)
    colors = random_colors(n)
    alphas = random_alpha(n)
    locations, strings, fonts, sizes, rotations, text_alphas = random_texts()
    for xyxy, color, alpha in zip(lines, colors, alphas):
        ad.line(xyxy, width=1.5, color=color, alpha=alpha)
    for rect, color, alpha in zip(rects, colors, alphas):
        ad.rect(rect, color=color, alpha=alpha)
    for x in range(0, ASIZE, 100):
        ad.draw_string((x, x), "Hello world!")
    for circle, color, alpha in zip(circles, colors, alphas):
        ad.circle(circle, color=color, alpha=alpha)
    for ellipse, color, alpha in zip(ellipses, colors, alphas):
        ad.ellipse(ellipse, color=color, alpha=alpha)
    for location, s, font, size, rotation, alpha in zip(locations, strings, fonts, sizes, rotations, text_alphas):
        ad.draw_string(location, s, font=font, alpha="mask", rotation=rotation, size=size)


def test_display_list():
    canvas, ad = setup()
    fill_display_list(ad, 100)
    with StopWatcher() as show:
        ad.render()
        a = ad.read()
        show(ad, a)


def test_chunking():
    times = []
    for chunk_size in (10, 100, 500):
        canvas, ad = setup(chunk_size=chunk_size)
        with StopWatcher() as show:
            t = time.time()
            fill_display_list(ad, 1000)
            ad.render()
            canvas = ad.read()
            times.append(time.time() - t)
            show(ad, f"{chunk_size}", canvas)
    print(times)
