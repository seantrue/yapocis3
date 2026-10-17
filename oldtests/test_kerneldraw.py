#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.
import time
import pytest
from typing import Optional

import numpy as np

from .random_data import (
    random_alpha,
    random_circles,
    random_colors,
    random_composites,
    random_ellipses,
    random_lines,
    random_rects,
    random_sprites,
    random_texts,
)
from yapocis.lib.kerneldraw1 import KernelDraw1
from yapocis.utils.stereo_typing import Image1

from . import ASIZE, StopWatcher


def just_canvas() -> Image1:
    canvas = np.zeros((ASIZE, ASIZE), dtype=np.float32)
    return canvas


def setup_test(device_name: str | None = None, chunk_size: int | None = None):
    canvas = just_canvas()
    return canvas, KernelDraw1(canvas, device_name=device_name, chunk_size=chunk_size)


def test_batch():
    canvas, ad = setup_test()
    locations, strings, fonts, sizes, _, text_alphas = random_texts()
    with StopWatcher("batch", kernel=ad) as sw:
        with sw.checkpoint("display_list"):
            ad.composite(*random_composites(), alpha=1.0)
            ad.sprites(*random_sprites(), alpha=1.0)
            ad.ellipses(random_ellipses(), colors=random_colors(), alphas=random_alpha())
            ad.circles(random_circles(), colors=random_colors(), alphas=random_alpha())
            ad.rects(random_rects(), colors=random_colors(), alphas=random_alpha())
            ad.lines(random_lines(), colors=random_colors(), alphas=random_alpha())
            ad.draw_strings(locations, strings, fonts=fonts, sizes=sizes, alphas=text_alphas)
        with sw.checkpoint("render"):
            ad.render()
        with sw.checkpoint("read"):
            a = ad.read()
        sw.show(a)


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
    canvas, ad = setup_test()
    fill_display_list(ad, 100)
    with StopWatcher("display list", kernel=ad) as sw:
        with sw.checkpoint("render"):
            ad.render()
        with sw.checkpoint("read"):
            a = ad.read()
        sw.show(a)

# @pytest.mark.skip("Too slow for now")
def test_chunking():
    times = []
    for chunk_size in (10, 100, 500):
        canvas, ad = setup_test(chunk_size=chunk_size)
        with StopWatcher(f"chunking {chunk_size}", kernel=ad) as sw:
            t = time.time()
            with sw.checkpoint("fill_display_list"):
                fill_display_list(ad, 1000)
            with sw.checkpoint("render"):
                ad.render()
            with sw.checkpoint("read"):
                canvas = ad.read()
            times.append(time.time() - t)
            sw.show(canvas, f"chunk_size={chunk_size}")
    print(times)
