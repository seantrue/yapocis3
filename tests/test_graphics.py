#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from random import randint

from yapocis.lib.channels import split_channels
from yapocis.lib.hsi import HSI
from yapocis.lib.graphics import Graphics1
from yapocis.utils.typing import *

from . import image_test, ASIZE, interesting_words, double_shuffle, show, snapshow, show_snapshots
hsis = HSI()

def setup(device_name:Optional[str]=None):
    return np.zeros((ASIZE, ASIZE)).astype(np.float32), Graphics1(device_name=device_name)

def image_source(align=5):
    image = image_test()
    r, g, b = split_channels(image)
    h, s, i = hsis.rgb2hsi(r, g, b)
    d = min(*i.shape)
    d = d - int(d % align);
    i = i[:d, :d]
    return i


def image_sprites(steps=4):
    image = image_source(align=steps)
    dimension = image.shape[0]
    step = dimension // steps
    blocks = []
    for x in range(0, dimension, step):
        for y in range(0, dimension, step):
            blocks.append(image[x:x + step, y:y + step])
    return blocks


def random_points(n=100):
    def rx():
        return randint(0, ASIZE)

    def ry():
        return randint(0, ASIZE)

    points = []
    for iline in range(n):
        x, y = rx(), ry()
        points.append([x, y])
    return points


def random_colors(n=100):
    return np.random.uniform(0, 1.0, n).astype(np.float32)


def random_alpha(n=100):
    return np.random.uniform(.25, 1.0, n).astype(np.float32)


def random_lines(n=100):
    lines = random_points(n)
    length = 50
    for line in lines:
        x, y = line
        x1, y1 = x + randint(-length, length), y + randint(-length, length)
        line += [x1, y1]
    return lines


random_rects = random_lines


def random_triangles(n=100):
    lines = random_points(n)
    a = np.zeros((ASIZE, ASIZE)).astype(np.float32)
    length = 50
    for line in lines:
        x, y = line
        x1, y1 = x + randint(-length, length), y + randint(-length, length)
        line += [x1, y1]
        x1, y1 = x + randint(-length, length), y + randint(-length, length)
        line += [x1, y1]
    return lines


def random_circles(n=100):
    circles = random_points(n)
    radius = 50
    for circle in circles:
        circle.append(randint(2, radius))
    return circles


def random_ellipses(n=100):
    ellipses = random_points(n=n)
    radius = 50
    for ellipse in ellipses:
        ellipse += [randint(2, radius), randint(2, radius)]
    return ellipses


def random_composites(n=100):
    locations = random_points(n)
    colors = random_colors(n)
    sprites = image_sprites()
    blocks = sprites[:]
    while len(blocks) < n:
        blocks += sprites
    blocks = blocks[:n]
    return locations, blocks


def random_sprites(n=100):
    random_coords, random_sprites = random_composites(n)
    random_sprites = random_sprites[:10]
    for i in range(len(random_coords)):
        j = int(i % len(random_sprites))
        random_coords[i].insert(0, j)
    return random_coords, random_sprites


def random_texts(n=100):
    locations = random_points(n)
    strings = interesting_words(n // 10)
    fonts = ["bold", "italic", "light", "regular"]
    sizes = [18, 36, 72, 112]
    alphas = ["bug", "mask", .5, 1.0]
    rotations = double_shuffle(list(range(0, 360, 30)), n)
    strings = double_shuffle(strings, n)
    sizes = double_shuffle(sizes, n)
    fonts = double_shuffle(fonts, n)
    alphas = double_shuffle(alphas, n)
    return locations, strings, fonts, sizes, rotations, alphas


def test_lines():
    canvas,grph = setup()
    random_coords = random_lines()
    coords = []
    for i in range(100, ASIZE, 100):
        for j in range(100, ASIZE, 100):
            coords.append([0, 0, i, j])
    grph.lines(canvas, coords, distance=1.5, colors=random_colors(len(coords)), alpha=1.0)
    grph.lines(canvas, random_coords, distance=1.5, colors=random_colors(), alphas=random_alpha())
    snapshow(grph, canvas)


def test_rects():
    canvas,grph = setup()
    random_coords = random_lines()
    coords = []
    for i in range(100, ASIZE, 100):
        for j in range(100, ASIZE, 100):
            coords.append([i - 25, j - 25, i + 25, j + 25])
    grph.rects(canvas, coords, color=1.0, alpha=0.5)
    grph.rects(canvas, random_coords, colors=random_colors(), alphas=random_alpha())
    snapshow(grph, canvas)

def test_circles():
    canvas,grph = setup()
    random_coords = random_circles()
    coords = []
    for i in range(100, ASIZE, 100):
        for j in range(100, ASIZE, 100):
            coords.append([i, j, 100])
    grph.circles(canvas, coords, color=1.0, alpha=0.5)
    grph.circles(canvas, random_coords, colors=random_colors(), alphas=random_alpha())
    snapshow(grph, canvas)


def test_triangles():
    canvas,grph = setup()
    random_coords = random_triangles()
    coords = []
    for i in range(100, ASIZE, 100):
        for j in range(100, ASIZE, 100):
            coords.append([i, j, i + 50, j + 0, i + 50, j - 50])
    grph.triangles(canvas, coords, color=1.0, alpha=0.5)
    grph.triangles(canvas, random_coords, colors=random_colors(), alphas=random_alpha())
    snapshow(grph, canvas)


def test_ellipses():
    canvas,grph = setup()
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
    grph.ellipses(canvas, coords, colors=colors, alpha=alpha)
    grph.ellipses(canvas, random_coords, colors=random_colors(), alphas=random_alpha())
    snapshow(grph, canvas)


def test_composite():
    canvas,grph = setup()
    random_coords, random_blocks = random_composites()
    grph.composite(canvas, random_coords, random_blocks)
    snapshow(grph, canvas)


def test_composite_sprites():
    canvas,grph = setup()
    random_coords, random_sprites_ = random_sprites()
    grph.sprites(canvas, random_coords, random_sprites_)
    snapshow(grph, canvas)


def test_composite_mosaic():
    _, grph = setup()
    image = image_test()
    from random import shuffle
    r, g, b = split_channels(image)
    h, s, i = hsis.rgb2hsi(r, g, b)
    d = min(*i.shape)
    d = d - int(d % 5);
    i = i[:d, :d]
    show("i", i)
    canvas = np.zeros_like(i)
    step = d // 5
    blocks = [i]
    coords = [(0, 0)]
    for x in range(0, d, step):
        for y in range(0, d, step):
            block = i[x:x + step, y:y + step]
            coords.append((x, y))
            blocks.append(block)
    shuffle(blocks)
    grph.composite(canvas, coords, blocks)
    snapshow(grph, canvas)


def test_simple_texts():
    canvas,grph = setup()
    locations = []
    text = interesting_words(100)
    for x in range(100, ASIZE, 100):
        locations.append((x // 2, x))
    strings = double_shuffle(text, len(locations))
    grph.draw_strings(canvas, locations, strings, font="default", size=32, alpha="bug")
    snapshow(grph, canvas)

def test_text_rotation_centered():
    canvas, grph = setup()
    rotations = list(range(-180,180,30))
    center = canvas.shape[0]//2
    locations = [(center,center)] * len(rotations)
    strings = [f"-------- {r:03d}" for r in rotations]
    grph.draw_strings(canvas, locations, strings, rotations=rotations, color=1, alpha="mask")
    grph.circles(canvas, [(center,center,20)],color=.5)
    snapshow(grph, canvas)

def test_text_rotation():
    canvas,grph = setup()
    rotations = list(range(0,360,10))
    center = canvas.shape[0]//2
    locations = [[center,center] for y in range(100,1000,800//len(rotations))]
    locations = locations[:len(rotations)]
    strings = [f"-------------------- {r:03d}" for r in rotations]
    grph.draw_strings(canvas, locations, strings, size=32, rotations=rotations, color=1, alpha="mask")
    for location in locations:
        location.append(10)
        location[0],location[1] = location[1],location[0]
    grph.circles(canvas, locations[:-1],color=.5)
    snapshow(grph, canvas)


def test_texts():
    canvas,grph = setup()
    locations, strings, fonts, sizes, rotations, alphas = random_texts(300)
    grph.draw_strings(canvas, locations, strings, fonts=fonts, sizes=sizes, alphas=alphas)
    snapshow(grph, canvas)

def test_stressed_texts():
    canvas,grph = setup()
    locations, strings, fonts, sizes, rotations, _ = random_texts(300)
    grph.draw_strings(canvas, locations, strings, fonts=fonts, sizes=sizes, rotations=rotations, alpha="mask")
    snapshow(grph, canvas)

def test_one():
    canvas, grph = setup()
    grph.lines(canvas, [[0,0,ASIZE,ASIZE]], distance=1.5, color=1.0, alpha=1.0)
    grph.circles(canvas, [[ASIZE/2,ASIZE/2, 20]],color=1.0, alpha=1.0)
    grph.rects(canvas, [[ASIZE/4,ASIZE/4, ASIZE/4+40, ASIZE/4+40]],color=1.0, alpha=1.0)
    grph.ellipses(canvas, [[ASIZE/4,ASIZE/2, 80, 40]],color=1.0, alpha=1.0)
    grph.triangles(canvas, [[ASIZE*3/4,ASIZE/2,40+ASIZE*3/4,20+ASIZE/2,40+ASIZE*3/4,-20+ASIZE/2 ]],color=1.0, alpha=1.0)
    grph.draw_strings(canvas, [[ASIZE*3/4,ASIZE*3/4]], ["Hello world"], font="bold", color=1.0, alpha="mask", size=32)
    snapshow(grph, canvas)


