#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.
import time
import numpy as np
import pytest

from tests.test_graphics import random_ellipses, random_circles, random_rects, random_lines, random_composites, \
    random_sprites, random_colors, random_alpha, random_texts
from yapocis.lib.kerneldraw import KernelDraw
from yapocis.utils.basic import show_array
from . import ASIZE

def just_canvas():
    canvas = np.zeros((ASIZE, ASIZE), dtype=np.float32)
    return canvas


def setup():
    canvas = just_canvas()
    return canvas, KernelDraw(canvas)


def show(label, a):
    show_array(f"kerneldraw: {label}", a)


def test_batch():
    canvas, ad = setup()
    locations, strings, fonts, sizes, _, text_alphas = random_texts()
    ad.composite(*random_composites(), alpha=1.0)
    ad.sprites(*random_sprites(), alpha=1.0)
    ad.ellipses(random_ellipses(), colors=random_colors(), alphas=random_alpha())
    ad.circles(random_circles(), colors=random_colors(), alphas=random_alpha())
    ad.rects(random_rects(), colors=random_colors(), alphas=random_alpha())
    ad.lines(random_lines(), colors=random_colors(), alphas=random_alpha())
    ad.draw_strings(locations, strings, fonts=fonts, sizes=sizes, alphas=text_alphas)
    ad.render()
    a = ad.read()
    show(f"batch", a)


def fill_displaylist(ad,n=1000):
    ellipses = random_ellipses(n)
    rects = random_rects(n)
    lines = random_lines(n)
    circles = random_circles(n)
    colors = random_colors(n)
    alphas = random_alpha(n)
    locations, strings, fonts, sizes, rotations, text_alphas = random_texts()
    for xyxy, color, alpha in zip(lines, colors, alphas):
        ad.line(xyxy, distance=1.5, color=color, alpha=alpha)
    for rect, color, alpha in zip(rects, colors, alphas):
        ad.rect(rect, color=color, alpha=alpha)
    for x in range(0, ASIZE, 100):
        ad.draw_string((x, x), "Hello world!")
    for circle, color, alpha in zip(circles, colors, alphas):
        ad.circle(circle, color=color, alpha=alpha)
    for ellipse, color, alpha in zip(ellipses, colors, alphas):
        ad.ellipse(ellipse, color=color, alpha=alpha)
    for location, s, font, size, rotation, alpha in zip(locations, strings, fonts, sizes, rotations, text_alphas):
        ad.draw_string(location, s, font=font, alpha="mask",rotation=rotation,size=size)


def test_displaylist():
    canvas, ad = setup()
    fill_displaylist(ad,100)
    ad.render()
    a = ad.read()
    show(f"displaylist", a)


def test_chunking():
    times = []
    for chunk_size in (None, 10, 100, 500, 10000):
        canvas = just_canvas()
        t = time.time()
        ad = KernelDraw(canvas, chunk_size=chunk_size)
        fill_displaylist(ad)
        ad.render()
        canvas = ad.read()
        times.append(time.time() - t)
        show(f"Chunked {chunk_size} {times[-1]:.1f}", canvas)
        del ad
    print(times)
