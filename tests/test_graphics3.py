#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.
import pytest
import numpy as np

from tests.test_graphics import random_ellipses, random_circles, random_rects, random_lines, random_triangles, \
    random_colors, random_alpha, ASIZE
from yapocis.basic import show_array
from yapocis.lib.channels import join_channels
from yapocis.lib.kerneldraw3 import KernelDraw3

ASIZE=1000

@pytest.fixture
def canvas():
    return np.zeros((ASIZE, ASIZE)).astype(np.float32)



def test_kerneldraw3_batch(canvas):
    random_coords = random_ellipses()
    canvas = join_channels(canvas, canvas, canvas)
    ad = KernelDraw3(canvas)
    circles = random_circles()
    rects = random_rects()
    lines = random_lines()
    triangles = random_triangles()
    ellipses = random_ellipses()
    alphas = random_colors()
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
    ad.render()
    a = ad.read()
    show_array(f"batch 3", canvas)


def test_kerneldraw3_displaylist(canvas):
    canvas[:,:] = 1.0
    canvas = join_channels(canvas, canvas, canvas)
    ad = KernelDraw3(canvas)
    circles = random_circles()
    rects = random_rects()
    lines = random_lines()
    colors = list(zip(random_colors(), random_colors(), random_colors()))
    alphas = random_alpha()
    for circle, color, alpha in zip(circles, colors, alphas):
        ad.circle(circle,color=color,alpha=alpha)
    for xyxy, color, alpha in zip(lines, colors, alphas):
        ad.line(xyxy, distance=1.5, color=color, alpha=alpha)
    for rect, color, alpha in zip(rects, colors, alphas):
        ad.rect(rect, color=color, alpha=alpha)
    for x in range(100, ASIZE, 100):
        ad.draw_string((x // 2, x), "Hello world!", alpha="bug")
    ad.render()
    a = ad.read()
    show_array(f"displaylist 3", canvas)