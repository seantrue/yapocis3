import numpy as np
from random import randint, shuffle

from yapocis.utils import show_array, imread
from yapocis.channels import join_channels, split_channels
from yapocis.lib.hsi import rgb2hsi
from yapocis.lib.kerneldraw import KernelDraw, KernelDraw3
from yapocis.lib.graphics import lines, rects, circles, triangles, ellipses, composite, sprites, draw_strings

from . import test_image_path

test_image = test_image_path()


def image_source(align=5):
    image = np.float32(imread(test_image))
    image /= 256.
    r, g, b = split_channels(image)
    h, s, i = rgb2hsi(r, g, b)
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
    a = np.zeros((1000, 1000)).astype(np.float32)

    def rx():
        return randint(0, a.shape[0])

    def ry():
        return randint(0, a.shape[1])

    a[:, :] = 0.0
    points = []
    for iline in range(n):
        x, y = rx(), ry()
        points.append([x, y])
    return a, points


def random_colors(n=100):
    return np.random.uniform(0, 1.0, n).astype(np.float32)


def random_lines(n=100):
    a, lines = random_points(n)
    a = np.zeros((1000, 1000)).astype(np.float32)
    length = 50
    for line in lines:
        x, y = line
        x1, y1 = x + randint(-length, length), y + randint(-length, length)
        line += [x1, y1]
    return a, lines


def random_triangles(n=100):
    a, lines = random_points(n)
    a = np.zeros((1000, 1000)).astype(np.float32)
    length = 50
    for line in lines:
        x, y = line
        x1, y1 = x + randint(-length, length), y + randint(-length, length)
        line += [x1, y1]
        x1, y1 = x + randint(-length, length), y + randint(-length, length)
        line += [x1, y1]
    return a, lines


def random_circles(n=100):
    a, circles = random_points(n)
    a = np.zeros((1000, 1000)).astype(np.float32)
    radius = 50
    for circle in circles:
        circle.append(randint(2, radius))
    return a, circles


def random_ellipses(n=100):
    a, ellipses = random_points(n=n)
    radius = 50
    for ellipse in ellipses:
        ellipse += [randint(2, radius), randint(2, radius)]
    return a, ellipses


def random_composites(n=100):
    a, locations = random_points(n)
    colors = random_colors(n)
    sprites = image_sprites()
    blocks = sprites[:]
    while len(blocks) < n:
        blocks += sprites
    blocks = blocks[:n]
    return a, locations, blocks


def random_sprites(n=100):
    a, random_coords, random_sprites = random_composites(n)
    random_sprites = random_sprites[:10]
    for i in range(len(random_coords)):
        j = int(i % len(random_sprites))
        random_coords[i].insert(0, j)
    return a, random_coords, random_sprites


def test_lines():
    a, random_coords = random_lines()
    coords = []
    for i in range(100, 1000, 100):
        for j in range(100, 1000, 100):
            coords.append([0, 0, i, j])
    lines(a, coords, 3.0, 1.0, .5)
    lines(a, random_coords, 1.5, random_colors(), random_colors())
    show_array(f"draw", a)


def test_rects():
    a, random_coords = random_lines()
    coords = []
    for i in range(100, 1000, 100):
        for j in range(100, 1000, 100):
            coords.append([i - 25, j - 25, i + 25, j + 25])
    rects(a, coords, 1.0, 0.5)
    rects(a, random_coords, random_colors(), random_colors())
    show_array(f"rects", a)


def test_circles():
    a, random_coords = random_circles()
    coords = []
    for i in range(100, 1000, 100):
        for j in range(100, 1000, 100):
            coords.append([i, j, 100])
    circles(a, coords, 1.0, 0.5)
    circles(a, random_coords, random_colors(), random_colors())
    show_array(f"circles", a)


def test_triangles():
    a, random_coords = random_triangles()
    coords = []
    for i in range(100, 1000, 100):
        for j in range(100, 1000, 100):
            coords.append([i, j, i + 50, j + 0, i + 50, j - 50])
    triangles(a, coords, 1.0, 0.5)
    triangles(a, random_coords, random_colors(), random_colors())
    show_array(f"triangles", a)


def test_ellipses():
    a, random_coords = random_ellipses()
    a = np.zeros((1000, 1000)).astype(np.float32)
    coords = []
    for i in range(100, 1000, 100):
        for j in range(100, 1000, 100):
            coords.append([i, j, 50, 100])
    ellipses(a, coords, 1.0, 0.5)
    coords = []
    for i in range(100, 1000, 100):
        coords.append([i, i, 100, 50])
    colors = np.linspace(0.0, 1.0, len(coords))
    opacity = 1.0
    ellipses(a, coords, colors, opacity)
    ellipses(a, random_coords, random_colors(), random_colors())
    show_array(f"ellipses", a)


def test_composite():
    a, random_coords, random_blocks = random_composites()
    composite(a, random_coords, random_blocks)
    show_array("composite", a)


def test_composite_sprites():
    a, random_coords, random_sprites_ = random_sprites()
    sprites(a, random_coords, random_sprites_)
    show_array("sprites", a)


def test_composite_mosaic():
    from random import shuffle
    image = np.float32(imread(test_image))
    image /= 256.
    r, g, b = split_channels(image)
    h, s, i = rgb2hsi(r, g, b)
    d = min(*i.shape)
    d = d - int(d % 5);
    i = i[:d, :d]
    show_array("i", i)
    iout = np.zeros_like(i)
    step = d // 5
    blocks = [i]
    coords = [(0, 0)]
    for x in range(0, d, step):
        for y in range(0, d, step):
            block = i[x:x + step, y:y + step]
            coords.append((x, y))
            blocks.append(block)
    shuffle(blocks)
    composite(iout, coords, blocks)
    show_array("mosaic", iout)


def test_simple_texts():
    a, _ = random_points()
    locations = []
    string = "Hello world"
    for x in range(100, 1000, 100):
        locations.append((x // 2, x))
    strings = ["string"] * len(locations)
    draw_strings(a, locations, strings, "default", 32, alphas=1.0)
    show_array("simple texts", a)


def test_texts():
    a, locations = random_points()
    strings = ["Hello", "world", "Goodbye", "cruel"]
    sizes = [18, 24, 32, 48]
    while len(strings) < len(locations):
        strings += strings
        sizes += sizes
    strings = strings[:len(locations)]
    sizes = sizes[:len(strings)]
    shuffle(strings)
    shuffle(sizes)
    draw_strings(a, locations, strings, "default", sizes, alphas="mask")
    show_array("texts", a)


def test_kerneldraw_batch():
    a, random_coords = random_ellipses()
    a[:] = 0.0
    ad = KernelDraw(a)
    _, circles = random_circles()
    _, rects = random_lines()
    _, lines = random_lines()
    _, locations, blocks = random_composites()
    _, sprite_locations, sprites = random_sprites()
    ad.ellipses(random_coords, random_colors(), random_colors())
    ad.circles(circles, random_colors(), random_colors())
    ad.rects(rects, random_colors(), random_colors())
    ad.lines(lines, 1.5, random_colors(), random_colors())
    ad.composite(locations, blocks, random_colors())
    ad.sprites(sprite_locations, sprites, alpha=1.0)
    ad.render()
    a = ad.read()
    show_array(f"batch", a)


def test_kerneldraw_displaylist():
    a, random_coords = random_ellipses()
    a[:] = 0.0
    ad = KernelDraw(a)
    _, circles = random_circles()
    _, rects = random_lines()
    _, lines = random_lines()
    for xyxy, color, alpha in zip(lines, random_colors(), random_colors()):
        ad.line(xyxy, 1.5, color, alpha)
    for rect, color, alpha in zip(rects, random_colors(), random_colors()):
        ad.rect(rect, color, alpha)
    for x in range(0, 1000, 100):
        ad.draw_string((x, x), "Hello world!")
    ad.render()
    a = ad.read()
    show_array(f"displaylist", a)


def test_kerneldraw3_batch():
    a, random_coords = random_ellipses()
    a = join_channels(a, a, a)
    ad = KernelDraw3(a)
    _, circles = random_circles()
    _, rects = random_lines()
    _, lines = random_lines()
    _, triangles = random_triangles()
    _, ellipses = random_ellipses()
    alpha = random_colors()
    ad.set_color((1, 0, 0))
    ad.ellipses(random_coords, alpha=alpha)
    ad.set_color((0, 1, 0))
    ad.circles(circles, alpha=alpha)
    ad.set_color((0, 0, 1))
    ad.rects(rects, alpha=alpha)
    ad.set_color((1, 1, 0))
    ad.lines(lines, distance=1, alpha=alpha)
    ad.set_color((0, 1, 1))
    ad.triangles(triangles, colors=(random_colors(), random_colors(), random_colors()), alpha=alpha)
    ad.set_color((1, 1, 1))
    ad.ellipses(ellipses, (.5, .5, .2), alpha=alpha)
    ad.render()
    a = ad.read()
    show_array(f"batch3", a)


def test_kerneldraw3_displaylist():
    a, random_coords = random_ellipses()
    a[:] = 0.0
    a = join_channels(a, a, a)
    ad = KernelDraw3(a)
    _, circles = random_circles()
    _, rects = random_lines()
    _, lines = random_lines()
    colors = list(zip(random_colors(), random_colors(), random_colors()))
    alphas = random_colors()
    for xyxy, color, alpha in zip(lines, colors, alphas):
        ad.line(xyxy, 1.5, color, alpha)
    for rect, color, alpha in zip(rects, colors, alphas):
        ad.rect(rect, color, alpha)
    for x in range(100, 1000, 100):
        ad.draw_string((x // 2, x), "Hello world!", alpha="mask")
    ad.render()
    a = ad.read()
    show_array(f"displaylist3", a)
