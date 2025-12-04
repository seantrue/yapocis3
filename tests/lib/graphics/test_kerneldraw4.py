#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

import time

import numpy as np
import pytest

from yapocis.core import get_test_backends
from yapocis.lib.kerneldraw4 import KernelDraw4
from yapocis.utils.stereo_typing import Image4

from ...random_data import random_alpha_4ch as random_alpha
from ...random_data import (
    random_circles,
)
from ...random_data import random_colors_4ch as random_colors
from ...random_data import random_composites_4ch as random_composites
from ...random_data import (
    random_ellipses,
    random_lines,
    random_rects,
)
from ...random_data import random_sprites_4ch as random_sprites
from . import ASIZE, StopWatcher


def just_canvas() -> Image4:
    return np.zeros((ASIZE, ASIZE, 4), dtype=np.float32)


def setup_test(
    device_name: str | None = None, chunk_size: int | None = None
) -> tuple[Image4, KernelDraw4]:
    canvas = just_canvas()
    return canvas, KernelDraw4(canvas, device_name=device_name, chunk_size=chunk_size)


@pytest.mark.parametrize("backend", get_test_backends())
def test_batch(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    canvas, ad = setup_test()
    random_coords = random_ellipses()
    circles = random_circles()
    rects = random_rects()
    lines = random_lines()
    locations, blocks = random_composites()
    sprite_locations, sprites = random_sprites()
    with StopWatcher("batch", kernel=ad) as sw:
        with sw.checkpoint("display_list"):
            ad.composite(locations, blocks)
            ad.sprites(sprite_locations, sprites)
            ad.ellipses(random_coords, colors=random_colors())
            ad.circles(circles, colors=random_colors())
            ad.rects(rects, colors=random_colors())
            ad.lines(lines, colors=random_colors(), distance=1.5)
        with sw.checkpoint("render"):
            ad.render()
        with sw.checkpoint("read"):
            a = ad.read()
        sw.show(a)


@pytest.mark.parametrize("backend", get_test_backends())
def test_displaylist(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    canvas, ad = setup_test()
    rects = random_rects()
    lines = random_lines()
    with StopWatcher("display list", kernel=ad) as sw:
        with sw.checkpoint("display_list"):
            for xyxy, color in zip(lines, random_colors(), strict=False):
                ad.line(xyxy, color=color, distance=1.5)
            for rect, color in zip(rects, random_colors(), strict=False):
                ad.rect(rect, color=color)
            for x in range(0, ASIZE, 100):
                ad.draw_string((x, x), "Hello world!", color=[1, 1, 1, 0.5])
        with sw.checkpoint("render"):
            ad.render()
        with sw.checkpoint("read"):
            a = ad.read()
        sw.show(a)


@pytest.mark.parametrize("backend", get_test_backends())
def test_chunking(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
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
        with StopWatcher(f"chunking {chunk_size}", kernel=ad) as sw:
            with sw.checkpoint("display_list"):
                for xyxy, color, alpha in zip(lines, colors, alphas, strict=False):
                    ad.line(xyxy, distance=1.5, color=color, alpha=alpha)
                for rect, color, alpha in zip(rects, colors, alphas, strict=False):
                    ad.rect(rect, color=color, alpha=alpha)
                for x in range(0, ASIZE, 100):
                    ad.draw_string((x, x), "Hello world!")
            with sw.checkpoint("render"):
                ad.render()
            with sw.checkpoint("read"):
                canvas = ad.read()
            times.append(time.time() - t)
            sw.show(canvas, f"chunk_size={chunk_size}")
    print(times)
