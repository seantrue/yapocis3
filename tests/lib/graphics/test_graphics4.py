#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.

import pytest

from yapocis.core import get_test_backends
from yapocis.lib.graphics4 import Graphics4
from yapocis.utils.stereo_typing import *

from ...random_data import (  # Graphics4-specific imports  # Graphics4-specific imports  # Graphics4-specific imports  # Graphics4-specific imports
    components_to_colors,
    random_circles,
    random_ellipses,
    random_icons,
    random_lines,
    random_points,
    random_texts,
    random_triangles,
)
from ...random_data import image_sprites_4ch as image_sprites
from ...random_data import random_alpha_4ch as random_alpha
from ...random_data import random_colors_4ch as random_colors
from ...random_data import random_composites_4ch as random_composites
from ...random_data import random_sprites_4ch as random_sprites
from . import ASIZE, StopWatcher, double_shuffle, text_test_lines

RED = [1, 0, 0, 1]
GREEN = [0, 1, 0, 1]
BLUE = [0, 0, 1, 1]
YELLOW = [0.7, 0.7, 0, 1]
WHITE = [1, 1, 1, 1]
PURPLE = [1, 0, 1, 1]


def setup_test(device_name=None):
    return np.zeros((ASIZE, ASIZE, 4)).astype(np.float32), Graphics4(device_name)


@pytest.mark.parametrize("backend", get_test_backends())
def test_alpha(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    canvas, grph = setup_test()
    with StopWatcher("alpha", kernel=grph) as sw:
        center = ASIZE // 2
        width = ASIZE // 6
        offset = ASIZE // 3
        with sw.checkpoint("alpha"):
            coords = [
                (0, offset, ASIZE - 1, offset * 2),
            ]
            grph.rects(canvas, coords, color=GREEN, alpha=0.5)
            coords = [(offset, 0, offset * 2, ASIZE - 1)]
            grph.rects(canvas, coords, color=RED, alpha=1.0)
        sw.show(canvas)


@pytest.mark.smoke
@pytest.mark.parametrize("backend", get_test_backends())
def test_lines(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    canvas, grph = setup_test()
    random_coords = random_lines()
    coords = []
    for i in range(100, ASIZE, 100):
        for j in range(100, ASIZE, 100):
            coords.append([0, 0, i, j])
    with StopWatcher("lines", kernel=grph) as sw:
        with sw.checkpoint("lines"):
            grph.lines(
                canvas, coords, width=1.5, colors=random_colors(len(coords)), alpha=1.0
            )
            grph.lines(
                canvas,
                random_coords,
                distance=1.5,
                colors=random_colors(),
                alphas=random_alpha(),
            )
        sw.show(canvas)


@pytest.mark.parametrize("backend", get_test_backends())
def test_rects(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    canvas, grph = setup_test()
    coords = []
    for i in range(100, ASIZE, 100):
        for j in range(100, ASIZE, 100):
            coords.append([i - 25, j - 25, i + 25, j + 25])
    colors = random_colors(len(coords))
    with StopWatcher("rects", kernel=grph) as sw:
        with sw.checkpoint("rects"):
            grph.rects(canvas, coords, colors=colors)
            grph.rects(canvas, random_lines(), colors=random_colors())
        sw.show(canvas)


@pytest.mark.parametrize("backend", get_test_backends())
def test_circles(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    canvas, grph = setup_test()
    random_coords = random_circles()
    coords = []
    for i in range(100, ASIZE, 100):
        for j in range(100, ASIZE, 100):
            coords.append([i, j, 100])
    colors = random_colors(len(coords))
    with StopWatcher("circles", kernel=grph) as sw:
        with sw.checkpoint("circles"):
            grph.circles(canvas, coords, colors=colors)
            grph.circles(canvas, random_coords, colors=random_colors())
        sw.show(canvas)


@pytest.mark.parametrize("backend", get_test_backends())
def test_triangles(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    canvas, grph = setup_test()
    random_coords = random_triangles()
    coords = []
    for i in range(100, ASIZE, 100):
        for j in range(100, ASIZE, 100):
            coords.append([i, j, i + 50, j + 0, i + 50, j - 50])
    colors = random_colors(len(coords))
    with StopWatcher("triangles", kernel=grph) as sw:
        with sw.checkpoint("triangles"):
            grph.triangles(canvas, coords, colors=colors)
            grph.triangles(canvas, random_coords, colors=random_colors())
        sw.show(canvas)


@pytest.mark.parametrize("backend", get_test_backends())
def test_ellipses(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    canvas, grph = setup_test()
    random_coords = random_ellipses()
    coords = []
    for i in range(100, ASIZE, 100):
        for j in range(100, ASIZE, 100):
            coords.append([i, j, 50, 100])
    colors = random_colors(len(coords))
    grph.ellipses(canvas, coords, colors=colors)
    coords = []
    for i in range(100, ASIZE, 100):
        coords.append([i, i, 100, 50])
    colors = np.linspace(0.0, 1.0, len(coords))
    colors = components_to_colors(colors, colors, colors, colors)
    with StopWatcher("ellipses", kernel=grph) as sw:
        with sw.checkpoint("ellipses"):
            grph.ellipses(canvas, coords, colors=colors)
            grph.ellipses(canvas, random_coords, colors=random_colors())
        sw.show(canvas)


@pytest.mark.parametrize("backend", get_test_backends())
def test_composite(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    canvas, grph = setup_test()
    random_coords, random_blocks = random_composites()
    with StopWatcher("composite", kernel=grph) as sw:
        with sw.checkpoint("composite"):
            grph.composite(canvas, random_coords, random_blocks)
        sw.show(canvas)


@pytest.mark.parametrize("backend", get_test_backends())
def test_composite_sprites(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    canvas, grph = setup_test()
    random_coords, random_sprites_ = random_sprites()
    with StopWatcher("sprites", kernel=grph) as sw:
        with sw.checkpoint("sprites"):
            grph.sprites(canvas, random_coords, random_sprites_)
        sw.show(canvas)


@pytest.mark.parametrize("backend", get_test_backends())
def test_composite_mosaic(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    from random import shuffle

    _, grph = setup_test()
    sprites = image_sprites(5)
    coords, blocks = zip(*sprites, strict=False)
    coords = list(coords)
    blocks = list(blocks)
    h, w, chan = blocks[0].shape
    d = h * 5
    canvas = np.zeros((d, d, chan), dtype=np.float32)
    shuffle(blocks)
    with StopWatcher("mosaic", kernel=grph) as sw:
        with sw.checkpoint("composite"):
            grph.composite(canvas, coords, blocks)
        sw.show(canvas)


@pytest.mark.parametrize("backend", get_test_backends())
def test_simple_icons(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    canvas, grph = setup_test()
    locations = []
    # canvas[:,:,:] = .5
    for x in range(100, ASIZE, 100):
        for y in range(100, ASIZE, 100):
            locations.append((x, y))
    blocks = random_icons(len(locations))
    colors = random_colors(len(locations))
    for block, color in zip(blocks, colors, strict=False):
        block[:, :, :] *= color
    with StopWatcher("simple icons", kernel=grph) as sw:
        with sw.checkpoint("composite"):
            grph.composite(canvas, locations, blocks)
        sw.show(canvas)


@pytest.mark.parametrize("backend", get_test_backends())
def test_random_icons(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    canvas, grph = setup_test()
    locations = random_points()
    blocks = random_icons(len(locations))
    colors = random_colors(len(locations))
    for block, color in zip(blocks, colors, strict=False):
        block[:, :, :] *= color
    with StopWatcher("random icons", kernel=grph) as sw:
        with sw.checkpoint("composite"):
            grph.composite(canvas, locations, blocks)
        sw.show(canvas)


@pytest.mark.parametrize("backend", get_test_backends())
def test_simple_texts(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    canvas, grph = setup_test()
    locations = []
    s = "Hello world"
    for x in range(100, ASIZE, 100):
        locations.append((x // 2, x))
    strings = [s] * len(locations)
    colors = random_colors(len(strings))
    with StopWatcher("simple texts", kernel=grph) as sw:
        with sw.checkpoint("strings"):
            grph.draw_strings(canvas, locations, strings, colors=colors)
        sw.show(canvas)


@pytest.mark.parametrize("backend", get_test_backends())
def test_text_rotation(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    canvas, grph = setup_test()
    rotations = list(range(0, 360, 30))
    center = canvas.shape[0] // 2
    locations = [(center, center)] * len(rotations)
    strings = [f"--------{r:03d}" for r in rotations]
    color = np.array([1, 1, 1, 1])
    with StopWatcher("text rotation", kernel=grph) as sw:
        with sw.checkpoint("strings"):
            grph.draw_strings(
                canvas,
                locations,
                strings,
                color=color,
                rotations=rotations,
                alpha="mask",
            )
        with sw.checkpoint("circles"):
            grph.circles(canvas, [(ASIZE // 2, ASIZE // 2, 20)], color=[1, 0, 0, 0.5])
        sw.show(canvas)


@pytest.mark.parametrize("backend", get_test_backends())
def test_texts_stressed(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    canvas, grph = setup_test()
    locations, strings, fonts, sizes, rotations, alphas = random_texts(300)
    with StopWatcher("stressed texts", kernel=grph) as sw:
        with sw.checkpoint("strings"):
            grph.draw_strings(
                canvas,
                locations,
                strings,
                colors=random_colors(300),
                fonts=fonts,
                sizes=sizes,
                rotations=rotations,
                alphas=alphas,
            )
        sw.show(canvas)


@pytest.mark.parametrize("backend", get_test_backends())
def test_texts_stressed_rotation(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    N = 300
    ANGLE_STEP = 30
    canvas, grph = setup_test()
    locations = random_points(N)
    strings = double_shuffle(["Hello", "sympathy", "goodbye", "Aqua"], N)
    fonts = double_shuffle(["bold", "italic", "light", "regular"], N)
    sizes = double_shuffle([18, 36, 72, 112], N)
    alpha = double_shuffle(["bug", "mask", "bug", "mask"], N)
    rotations = double_shuffle(list(range(0, 360, ANGLE_STEP)), N)
    with StopWatcher("stressed rotated", kernel=grph) as sw:
        with sw.checkpoint("strings"):
            grph.draw_strings(
                canvas,
                locations,
                strings,
                colors=random_colors(len(locations)),
                fonts=fonts,
                sizes=sizes,
                rotations=rotations,
                alphas=alpha,
            )
        sw.show(canvas)


@pytest.mark.parametrize("backend", get_test_backends())
def test_typesetting(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    canvas, grph = setup_test()
    lines = text_test_lines()
    x, y = 60, 60
    locations = [(x, y + i * 30) for i in range(len(lines)) if lines[i]]
    lines = [line for line in lines if line]
    sizes = [24] * len(lines)
    sizes[0] = 48
    colors = [None] * len(lines)
    colors[0] = [0.0, 1.0, 0.1, 0]
    for i in range(len(lines)):
        if "Tyger" in lines[i]:
            colors[i] = [1.0, 0.1, 0, 0.1]
    with StopWatcher("typesetting", kernel=grph) as sw:
        with sw.checkpoint("strings"):
            grph.draw_strings(canvas, locations, lines, sizes=sizes, colors=colors)
        sw.show(canvas)


@pytest.mark.parametrize("backend", get_test_backends())
def test_one(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    canvas, grph = setup_test()
    with StopWatcher("one", kernel=grph) as sw:
        with sw.checkpoint("lines"):
            grph.lines(
                canvas, [[0, 0, ASIZE, ASIZE]], distance=1.5, color=RED, alpha=1.0
            )
        with sw.checkpoint("circles"):
            grph.circles(canvas, [[ASIZE / 2, ASIZE / 2, 20]], color=GREEN, alpha=1.0)
        with sw.checkpoint("rects"):
            grph.rects(
                canvas,
                [[ASIZE / 4, ASIZE / 4, ASIZE / 4 + 40, ASIZE / 4 + 40]],
                color=BLUE,
                alpha=1.0,
            )
        with sw.checkpoint("ellipses"):
            grph.ellipses(
                canvas, [[ASIZE / 4, ASIZE / 2, 80, 40]], color=YELLOW, alpha=1.0
            )
        with sw.checkpoint("triangles"):
            grph.triangles(
                canvas,
                [
                    [
                        ASIZE * 3 / 4,
                        ASIZE / 2,
                        40 + ASIZE * 3 / 4,
                        20 + ASIZE / 2,
                        40 + ASIZE * 3 / 4,
                        -20 + ASIZE / 2,
                    ]
                ],
                color=WHITE,
                alpha=1.0,
            )
        with sw.checkpoint("strings"):
            grph.draw_strings(
                canvas,
                [[ASIZE * 3 / 4, ASIZE * 3 / 4]],
                ["Hello world"],
                font="bold",
                color=PURPLE,
                alpha="mask",
                size=32,
            )
        sw.show(canvas)
