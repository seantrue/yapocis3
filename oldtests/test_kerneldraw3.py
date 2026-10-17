#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.
import numpy as np
from typing import Optional

from .random_data import (
    random_alpha,
    random_circles,
    random_colors,
    random_ellipses,
    random_lines,
    random_rects,
    random_texts,
    random_triangles,
)
from yapocis.lib.kerneldraw3 import KernelDraw3
from yapocis.utils.stereo_typing import Image3

from . import ASIZE, StopWatcher, show_snapshots, snapshow


def setup_test(device_name: str | None = None, chunk_size: int | None = None) -> tuple[Image3, KernelDraw3]:
    device_name = device_name if isinstance(device_name,str) else None
    canvas = np.zeros((ASIZE, ASIZE, 3), dtype=np.float32)
    return canvas, KernelDraw3(canvas, chunk_size=chunk_size, device_name=device_name)


def test_batch():
    canvas, ad = setup_test()
    locations, strings, fonts, sizes, rotations, text_alphas = random_texts()
    random_coords = random_ellipses()
    circles = random_circles()
    rects = random_rects()
    lines = random_lines()
    triangles = random_triangles()
    ellipses = random_ellipses()
    alphas = random_colors()
    with StopWatcher("batch", kernel=ad) as sw:
        with sw.checkpoint("display_list"):
            ad.set_color((1, 0, 0))
            ad.ellipses(random_coords, alphas=alphas)
            ad.set_color((0, 1, 0))
            ad.circles(circles, alphas=alphas)
            ad.set_color((0, 0, 1))
            ad.rects(rects, alphas=alphas)
            ad.set_color((1, 1, 0))
            ad.lines(lines, distance=1, alphas=alphas)
            ad.set_color((0, 1, 1))
            colors = random_colors(), random_colors(), random_colors()
            colors = np.column_stack(colors)
            ad.triangles(triangles, colors=colors, alphas=alphas)
            ad.set_color((1, 1, 1))
            ad.ellipses(ellipses, color=(.5, .5, .2), alphas=alphas)
            ad.draw_strings(locations, strings, fonts=fonts, sizes=sizes, rotations=rotations, colors=colors, alpha="mask")
        with sw.checkpoint("render"):
            ad.render()
        with sw.checkpoint("read"):
            a = ad.read()
        sw.show(a)


def test_displaylist():
    canvas, ad = setup_test()
    circles = random_circles()
    rects = random_rects()
    lines = random_lines()
    colors = list(zip(random_colors(), random_colors(), random_colors()))
    alphas = random_alpha()
    with StopWatcher("display list", kernel=ad) as sw:
        with sw.checkpoint("display_list"):
            for circle, color, alpha in zip(circles, colors, alphas):
                ad.circle(circle, color=color, alpha=alpha)
            for xyxy, color, alpha in zip(lines, colors, alphas):
                ad.line(xyxy, distance=1.5, color=color, alpha=alpha)
            for rect, color, alpha in zip(rects, colors, alphas):
                ad.rect(rect, color=color, alpha=alpha)
            for x in range(100, ASIZE, 100):
                ad.draw_string((x // 2, x), "Hello world!", alpha="bug")
        with sw.checkpoint("render"):
            ad.render()
        with sw.checkpoint("read"):
            a = ad.read()
        sw.show(a)
