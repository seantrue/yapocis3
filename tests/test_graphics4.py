#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import pytest

import numpy as np
from random import randint, shuffle

from yapocis.basic import show_array, imread
from yapocis.lib.kerneldraw4 import KernelDraw4
from yapocis.lib.graphics4 import lines, rects, circles, triangles, ellipses, composite, sprites, draw_strings

from . import test_image_path

test_image = test_image_path()

ASIZE=1000

@pytest.fixture
def canvas():
    return np.zeros((ASIZE, ASIZE, 4)).astype(np.float32)


def image_source(align=5):
    image = np.float32(imread(test_image))
    image /= 256.
    d = min(image.shape[0],image.shape[1])
    d = d - int(d % align);
    image4 = np.zeros((d,d,4), dtype=np.float32)
    image4[:,:,:3] = image[:d, :d,:]
    image4[:,:,3] = 1.0
    return image4


def image_sprites(steps=4):
    image = image_source(align=steps)
    dimension = image.shape[0]
    step = dimension // steps
    sprites = []
    for x in range(0, dimension, step):
        for y in range(0, dimension, step):
            sprites.append(((x,y),image[x:x + step, y:y + step,:]))
    return sprites


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

def random_vector(n=100):
    return np.random.uniform(0, 1.0, n).astype(np.float32)

def components_to_colors(*components):
    colors = list(zip(*components))
    colors = np.array(colors)
    return colors

def random_colors(n=100):
    colors = components_to_colors(random_vector(n),random_vector(n),random_vector(n),random_vector(n))
    return colors

def random_alpha(n=100):
    return random_vector(n)


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
    sprites = [sprite for _,sprite in image_sprites()]
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


def test_lines(canvas):
    random_coords = random_lines()
    coords = []
    for i in range(100, ASIZE, 100):
        for j in range(100, ASIZE, 100):
            coords.append([0, 0, i, j])
    colors = random_colors(len(coords))
    lines(canvas, coords, colors, width=1.5)
    lines(canvas, random_coords, random_colors(), width=1.5)
    show_array(f"draw", canvas)


def test_rects(canvas):
    coords = []
    for i in range(100, ASIZE, 100):
        for j in range(100, ASIZE, 100):
            coords.append([i - 25, j - 25, i + 25, j + 25])
    colors = random_colors(len(coords))
    rects(canvas, coords, colors)
    rects(canvas, random_lines(), random_colors())
    show_array(f"rects", canvas)


def test_circles(canvas):
    random_coords = random_circles()
    coords = []
    for i in range(100, ASIZE, 100):
        for j in range(100, ASIZE, 100):
            coords.append([i, j, 100])
    colors = random_colors((len(coords)))
    circles(canvas, coords, colors)
    circles(canvas, random_coords, random_colors())
    show_array(f"circles", canvas)


def test_triangles(canvas):
    random_coords = random_triangles()
    coords = []
    for i in range(100, ASIZE, 100):
        for j in range(100, ASIZE, 100):
            coords.append([i, j, i + 50, j + 0, i + 50, j - 50])
    colors = random_colors((len(coords)))
    triangles(canvas, coords, colors)
    triangles(canvas, random_coords, random_colors())
    show_array(f"triangles", canvas)


def test_ellipses(canvas):
    random_coords = random_ellipses()
    coords = []
    for i in range(100, ASIZE, 100):
        for j in range(100, ASIZE, 100):
            coords.append([i, j, 50, 100])
    colors = random_colors((len(coords)))
    ellipses(canvas, coords, colors)
    coords = []
    for i in range(100, ASIZE, 100):
        coords.append([i, i, 100, 50])
    colors = np.linspace(0.0, 1.0, len(coords))
    colors = components_to_colors(colors,colors,colors,colors)
    ellipses(canvas, coords, colors)
    ellipses(canvas, random_coords, random_colors())
    show_array(f"ellipses", canvas)


def test_composite(canvas):
    random_coords, random_blocks = random_composites()
    composite(canvas, random_coords, random_blocks)
    show_array("composite", canvas)


def test_composite_sprites(canvas):
    random_coords, random_sprites_ = random_sprites()
    sprites(canvas, random_coords, random_sprites_)
    show_array("sprites", canvas)


def test_composite_mosaic(canvas):
    from random import shuffle
    sprites = image_sprites(5)
    coords, blocks = zip(*sprites)
    coords = list(coords)
    blocks = list(blocks)
    h,w,chan = blocks[0].shape
    d = h*5
    iout = np.zeros((d,d,chan),dtype=np.float32)
    shuffle(blocks)
    composite(iout, coords, blocks)
    show_array("mosaic", iout)


def test_simple_texts(canvas):
    locations = []
    string = "Hello world"
    for x in range(100, ASIZE, 100):
        locations.append((x // 2, x))
    strings = ["string"] * len(locations)
    colors = random_colors(len(strings))
    draw_strings(canvas, locations, strings, colors=colors)
    show_array("simple texts", canvas)

def test_rotated_texts(canvas):
    locations = []
    string = "Hello world"
    rotations = list(range(0,360,30))
    locations = [(500,500)] * len(rotations)
    strings = [f"--------{r:03d}" for r in rotations]
    color = np.array([1,1,1,1])
    draw_strings(canvas, locations, strings, color=color, rotations=rotations, alpha="mask")
    circles(canvas, [(500,500,20)],[1,0,0,.5])
    show_array("simple texts", canvas)



def test_texts(canvas):
    locations = random_points(300)
    strings = ["Hello", "sympathy", "goodbye", "Aqua"]
    fonts = ["bold","italic","light","regular"]
    sizes = [18, 36, 72, 112]
    rotations = list(range(0,360,30))
    alpha = ["bug","mask","bug","mask"]
    while len(strings) < len(locations):
        strings += strings
        sizes += sizes
        fonts += fonts
        alpha += alpha
        rotations += rotations
    shuffle(strings)
    shuffle(sizes)
    shuffle(fonts)
    shuffle(alpha)
    shuffle(rotations)
    strings = strings[:len(locations)]
    sizes = sizes[:len(strings)]
    fonts = fonts[:len(sizes)]
    alpha = alpha[:len(sizes)]
    rotations = rotations[:len(sizes)]
    shuffle(alpha)
    draw_strings(canvas, locations, strings, colors=random_colors(len(locations)), fonts=fonts, sizes=sizes, rotations=rotations, alphas=alpha)
    show_array("texts", canvas)


def test_kerneldraw4_batch(canvas):
    random_coords = random_ellipses()
    ad = KernelDraw4(canvas)
    circles = random_circles()
    rects = random_rects()
    lines = random_lines()
    locations, blocks = random_composites()
    sprite_locations, sprites = random_sprites()
    ad.ellipses(random_coords, colors=random_colors())
    ad.circles(circles, colors=random_colors())
    ad.rects(rects, colors=random_colors())
    ad.lines(lines, colors=random_colors(), distance=1.5)
    ad.composite(locations, blocks)
    ad.sprites(sprite_locations, sprites)
    ad.render()
    a = ad.read()
    show_array(f"batch", canvas)


def test_kerneldraw4_displaylist(canvas):
    random_coords = random_ellipses()
    ad = KernelDraw4(canvas)
    circles = random_circles()
    rects = random_rects()
    lines = random_lines()
    for xyxy, color in zip(lines, random_colors()):
        ad.line(xyxy, color=color, distance=1.5)
    for rect, color in zip(rects, random_colors()):
        ad.rect(rect, color=color)
    for x in range(0, ASIZE, 100):
        ad.draw_string((x, x), "Hello world!", color=[1,1,1,.5])
    ad.render()
    a = ad.read()
    show_array(f"displaylist", canvas)


