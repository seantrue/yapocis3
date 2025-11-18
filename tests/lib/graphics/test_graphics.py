#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

import pytest
from yapocis.lib.graphics import Graphics1
from yapocis.utils.stereo_typing import *
from . import (
    ASIZE,
    StopWatcher,
    show,
)
from ...random_data import (
    random_points,
    random_colors,
    random_alpha,
    random_lines,
    random_widths,
    random_rects,
    random_triangles,
    random_circles,
    random_ellipses,
    random_composites,
    random_sprites,
    random_texts,
    image_sprites,
)

def setup_test(device_name: str | None = None) -> tuple[Image1, Graphics1]:
    return np.zeros((ASIZE, ASIZE)).astype(np.float32), Graphics1(
        device_name=device_name
    )

@pytest.mark.smoke
def test_lines():
    canvas, grph = setup_test()
    random_coords = random_lines()
    widths = random_widths()
    coords = []
    for i in range(100, ASIZE, 100):
        for j in range(100, ASIZE, 100):
            coords.append([0, 0, i, j])
    with StopWatcher("lines", kernel=grph) as sw:
        with sw.checkpoint("regular"):
            grph.lines(
                canvas, coords, width=1.0, colors=random_colors(len(coords)), alpha=1.0
            )
        with sw.checkpoint("random"):
            grph.lines(
                canvas,
                random_coords,
                widths=widths,
                colors=random_colors(len(random_coords)),
                alpha=1.0,
            )
        sw.show(canvas)

def test_rects():
    canvas, grph = setup_test()
    random_coords = random_lines()
    coords = []
    for i in range(100, ASIZE, 100):
        for j in range(100, ASIZE, 100):
            coords.append([i - 25, j - 25, i + 25, j + 25])
    with StopWatcher("rects", kernel=grph) as sw:
        with sw.checkpoint("regular"):
            grph.rects(canvas, coords, color=1.0, alpha=0.5)
        with sw.checkpoint("random"):
            grph.rects(
                canvas, random_coords, colors=random_colors(), alphas=random_alpha()
            )
        sw.show(canvas)

def test_circles():
    canvas, grph = setup_test()
    random_coords = random_circles()
    coords = []
    for i in range(100, ASIZE, 100):
        for j in range(100, ASIZE, 100):
            coords.append([i, j, 100])
    with StopWatcher("circles", kernel=grph) as sw:
        grph.circles(canvas, coords, color=1.0, alpha=0.5)
        grph.circles(
            canvas, random_coords, colors=random_colors(), alphas=random_alpha()
        )
        sw.checkpoint("circles")
        sw.show(canvas)

def test_triangles():
    canvas, grph = setup_test()
    random_coords = random_triangles()
    coords = []
    for i in range(100, ASIZE, 100):
        for j in range(100, ASIZE, 100):
            coords.append([i, j, i + 50, j + 0, i + 50, j - 50])
    with StopWatcher("triangles", kernel=grph) as sw:
        with sw.checkpoint("regular"):
            grph.triangles(canvas, coords, color=1.0, alpha=0.5)
        with sw.checkpoint("random"):
            grph.triangles(
                canvas, random_coords, colors=random_colors(), alphas=random_alpha()
            )
        sw.show(canvas)

def test_ellipses():
    canvas, grph = setup_test()
    random_coords = random_ellipses()
    coords = []
    for i in range(100, ASIZE, 100):
        for j in range(100, ASIZE, 100):
            coords.append([i, j, 50, 100])
    grph.ellipses(canvas, coords, color=1.0, alpha=0.5)
    coords = []
    for i in range(100, ASIZE, 100):
        coords.append([i, i, 100, 50])
    colors = np.linspace(0.0, 1.0, len(coords))
    alpha = 1.0
    with StopWatcher("ellipses", kernel=grph) as sw:
        with sw.checkpoint("regular"):
            grph.ellipses(canvas, coords, colors=colors, alpha=alpha)
        with sw.checkpoint("random"):
            grph.ellipses(
                canvas, random_coords, colors=random_colors(), alphas=random_alpha()
            )
        sw.show(canvas)

def test_composite():
    canvas, grph = setup_test()
    random_coords, random_blocks = random_composites()
    with StopWatcher("composite", kernel=grph) as sw:
        with sw.checkpoint("composite"):
            grph.composite(canvas, random_coords, random_blocks)
        sw.show(canvas)

def test_composite_sprites():
    canvas, grph = setup_test()
    random_coords, random_sprites_ = random_sprites()
    with StopWatcher("sprites", kernel=grph) as sw:
        with sw.checkpoint("random"):
            grph.sprites(canvas, random_coords, random_sprites_)
        sw.show(canvas)

def test_composite_mosaic():
    from random import shuffle
    from ...random_data import image_source_1ch

    _, grph = setup_test()
    i = image_source_1ch(align=5)
    show("i", i)
    canvas = np.zeros_like(i)
    d = i.shape[0]
    step = d // 5
    blocks = [i]
    coords = [(0, 0)]
    for x in range(0, d, step):
        for y in range(0, d, step):
            block = i[x : x + step, y : y + step]
            coords.append((x, y))
            blocks.append(block)
    shuffle(blocks)
    with StopWatcher("mosaic", kernel=grph) as sw:
        with sw.checkpoint("composite"):
            grph.composite(canvas, coords, blocks)
        sw.show(canvas)

def test_simple_texts():
    from . import interesting_words, double_shuffle

    canvas, grph = setup_test()
    locations = []
    text = interesting_words(100)
    for x in range(100, ASIZE, 100):
        locations.append((x // 2, x))
    strings = double_shuffle(text, len(locations))
    with StopWatcher("simple", kernel=grph) as sw:
        with sw.checkpoint("text"):
            grph.draw_strings(
                canvas, locations, strings, font="default", size=32, alpha="bug"
            )
        sw.show(canvas)

def test_text_rotation_centered():
    canvas, grph = setup_test()
    rotations = list(range(-180, 180, 30))
    center = canvas.shape[0] // 2
    locations = [(center, center)] * len(rotations)
    strings = [f"-------- {r:03d}" for r in rotations]
    with StopWatcher("center rotated", kernel=grph) as sw:
        with sw.checkpoint("text"):
            grph.draw_strings(
                canvas, locations, strings, rotations=rotations, color=1, alpha="mask"
            )
        with sw.checkpoint("circles"):
            grph.circles(canvas, [(center, center, 20)], color=0.5)
    sw.show(canvas)

def test_text_rotation():
    canvas, grph = setup_test()
    rotations = list(range(0, 360, 10))
    center = canvas.shape[0] // 2
    locations = [[center, center] for y in range(100, 1000, 800 // len(rotations))]
    locations = locations[: len(rotations)]
    strings = [f"-------------------- {r:03d}" for r in rotations]
    with StopWatcher("rotated", kernel=grph) as sw:
        with sw.checkpoint("text"):
            grph.draw_strings(
                canvas,
                locations,
                strings,
                size=32,
                rotations=rotations,
                color=1,
                alpha="mask",
            )
        sw.checkpoint("strings")
        for location in locations:
            location.append(10)
            location[0], location[1] = location[1], location[0]
        with sw.checkpoint("circles"):
            grph.circles(canvas, locations[:-1], color=0.5)
        sw.show(canvas)

def test_texts():
    canvas, grph = setup_test()
    locations, strings, fonts, sizes, rotations, alphas = random_texts(300)
    with StopWatcher("text", kernel=grph) as sw:
        with sw.checkpoint("strings"):
            grph.draw_strings(
                canvas, locations, strings, fonts=fonts, sizes=sizes, alphas=alphas
            )
        sw.show(canvas)

def test_stressed_texts():
    canvas, grph = setup_test()
    locations, strings, fonts, sizes, rotations, _ = random_texts(300)
    with StopWatcher("stressed text", kernel=grph) as sw:
        with sw.checkpoint("strings"):
            grph.draw_strings(
                canvas,
                locations,
                strings,
                fonts=fonts,
                sizes=sizes,
                rotations=rotations,
                alpha="mask",
            )
        sw.show(canvas)

def test_one():
    canvas, grph = setup_test()
    with StopWatcher("one", kernel=grph) as sw:
        with sw.checkpoint("all"):
            grph.lines(
                canvas, [[0, 0, ASIZE, ASIZE]], distance=1.5, color=1.0, alpha=1.0
            )
            grph.circles(canvas, [[ASIZE / 2, ASIZE / 2, 20]], color=1.0, alpha=1.0)
            grph.rects(
                canvas,
                [[ASIZE / 4, ASIZE / 4, ASIZE / 4 + 40, ASIZE / 4 + 40]],
                color=1.0,
                alpha=1.0,
            )
            grph.ellipses(
                canvas, [[ASIZE / 4, ASIZE / 2, 80, 40]], color=1.0, alpha=1.0
            )
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
                color=1.0,
                alpha=1.0,
            )
            grph.draw_strings(
                canvas,
                [[ASIZE * 3 / 4, ASIZE * 3 / 4]],
                ["Hello world"],
                font="bold",
                color=1.0,
                alpha="mask",
                size=32,
            )
        sw.show(canvas)
