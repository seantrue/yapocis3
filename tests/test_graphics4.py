#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from random import randint, shuffle

from yapocis.lib.graphics4 import Graphics4
from tests.test_graphics import random_texts
from yapocis.icons import icons, icon
from yapocis.utils.stereo_typing import *

from . import image_test, text_test_lines, ASIZE, double_shuffle, snapshow, show_snapshots

RED=[1,0,0,1]
GREEN=[0,1,0,1]
BLUE=[0,0,1,1]
YELLOW=[.7,.7,0,1]
WHITE=[1,1,1,1]
PURPLE=[1,0,1,1]
def setup(device_name=None):
    return np.zeros((ASIZE, ASIZE, 4)).astype(np.float32), Graphics4(device_name)

def image_source(align=5):
    image = image_test()
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

def random_icons(n=100):

    icon_names = icons()
    shuffle(icon_names)
    icon_names = icon_names[:n]
    colors = random_colors(n)
    icns = [icon(icon_name,size=100) for icon_name in icon_names]
    for icon_name, icn, color in zip(icon_names, icns, colors):
        if "essential" in icon_name:
            icn[:,:,:3] = (1.0 - icn[:,:,:3])
            # icn[:,:,:] *= color
            # icn[:,:,3] = 1.0
            # icn[:,:,:] = (1.0-icn[:,:,:])
            pass
    icns = double_shuffle(icns,n)
    return icns


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
    return np.random.uniform(.2, 1.0, n).astype(np.float32)

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
    sprites = [sprite for _,sprite in image_sprites(8)]
    blocks = sprites[:]
    blocks = double_shuffle(blocks, n)
    return locations, blocks


def random_sprites(n=100):
    random_coords, random_sprites = random_composites(n)
    random_sprites = random_sprites[:n]
    for i in range(len(random_coords)):
        j = int(i % len(random_sprites))
        random_coords[i].insert(0, j)
    return random_coords, random_sprites

def test_lines():
    canvas,grph = setup()
    random_coords = random_lines()
    coords = []
    for i in range(100, ASIZE, 100):
        for j in range(100, ASIZE, 100):
            coords.append([0, 0, i, j])
    grph.lines(canvas, coords, width=1.5, colors=random_colors(len(coords)), alpha=1.0)
    grph.lines(canvas, random_coords, distance=1.5, colors=random_colors(), alphas=random_alpha())
    snapshow(grph, canvas)


def test_rects():
    canvas, grph = setup()
    coords = []
    for i in range(100, ASIZE, 100):
        for j in range(100, ASIZE, 100):
            coords.append([i - 25, j - 25, i + 25, j + 25])
    colors = random_colors(len(coords))
    grph.rects(canvas, coords, colors=colors)
    grph.rects(canvas, random_lines(), colors=random_colors())
    snapshow(grph, canvas)


def test_circles():
    canvas, grph = setup()
    random_coords = random_circles()
    coords = []
    for i in range(100, ASIZE, 100):
        for j in range(100, ASIZE, 100):
            coords.append([i, j, 100])
    colors = random_colors((len(coords)))
    grph.circles(canvas, coords, colors=colors)
    grph.circles(canvas, random_coords, colors=random_colors())
    snapshow(grph, canvas)


def test_triangles():
    canvas, grph = setup()
    random_coords = random_triangles()
    coords = []
    for i in range(100, ASIZE, 100):
        for j in range(100, ASIZE, 100):
            coords.append([i, j, i + 50, j + 0, i + 50, j - 50])
    colors = random_colors((len(coords)))
    grph.triangles(canvas, coords, colors = colors)
    grph.triangles(canvas, random_coords, colors=random_colors())
    snapshow(grph, canvas)


def test_ellipses():
    canvas, grph = setup()
    random_coords = random_ellipses()
    coords = []
    for i in range(100, ASIZE, 100):
        for j in range(100, ASIZE, 100):
            coords.append([i, j, 50, 100])
    colors = random_colors((len(coords)))
    grph.ellipses(canvas, coords, colors=colors)
    coords = []
    for i in range(100, ASIZE, 100):
        coords.append([i, i, 100, 50])
    colors = np.linspace(0.0, 1.0, len(coords))
    colors = components_to_colors(colors,colors,colors,colors)
    grph.ellipses(canvas, coords, colors=colors)
    grph.ellipses(canvas, random_coords, colors=random_colors())
    snapshow(grph, canvas)


def test_composite():
    canvas, grph = setup()
    random_coords, random_blocks = random_composites()
    grph.composite(canvas, random_coords, random_blocks)
    snapshow(grph, canvas)


def test_composite_sprites():
    canvas, grph = setup()
    random_coords, random_sprites_ = random_sprites()
    grph.sprites(canvas, random_coords, random_sprites_)
    snapshow(grph, canvas)


def test_composite_mosaic():
    _, grph = setup()
    from random import shuffle
    sprites = image_sprites(5)
    coords, blocks = zip(*sprites)
    coords = list(coords)
    blocks = list(blocks)
    h,w,chan = blocks[0].shape
    d = h*5
    canvas = np.zeros((d,d,chan),dtype=np.float32)
    shuffle(blocks)
    grph.composite(canvas, coords, blocks)
    snapshow(grph, canvas)

def test_simple_icons():
    canvas, grph = setup()
    locations = []
    # canvas[:,:,:] = .5
    for x in range(100, ASIZE, 100):
        for y in range(100, ASIZE, 100):
            locations.append((x,y))
    blocks = random_icons(len(locations))
    colors = random_colors(len(locations))
    for block, color in zip(blocks,colors):
        block[:,:,:] *= color
    grph.composite(canvas, locations, blocks)
    snapshow(grph, canvas)

def test_random_icons():
    canvas, grph = setup()
    locations = random_points()
    blocks = random_icons(len(locations))
    colors = random_colors(len(locations))
    for block, color in zip(blocks,colors):
        block[:,:,:] *= color
    grph.composite(canvas, locations, blocks)
    snapshow(grph, canvas)


def test_simple_texts():
    canvas, grph = setup()
    locations = []
    s = "Hello world"
    for x in range(100, ASIZE, 100):
        locations.append((x // 2, x))
    strings = [s] * len(locations)
    colors = random_colors(len(strings))
    grph.draw_strings(canvas, locations, strings, colors=colors)
    snapshow(grph, canvas)

def test_text_rotation():
    canvas, grph = setup()
    rotations = list(range(0,360,30))
    center = canvas.shape[0]//2
    locations = [(center,center)] * len(rotations)
    strings = [f"--------{r:03d}" for r in rotations]
    color = np.array([1,1,1,1])
    grph.draw_strings(canvas, locations, strings, color=color, rotations=rotations, alpha="mask")
    grph.circles(canvas, [(ASIZE//2,ASIZE//2,20)],color=[1,0,0,.5])
    snapshow(grph, canvas)

def test_texts_stressed():
    canvas, grph = setup()
    locations, strings, fonts, sizes, rotations, alphas = random_texts(300)
    grph.draw_strings(canvas, locations, strings, colors=random_colors(300), fonts=fonts, sizes=sizes, rotations=rotations, alphas=alphas)
    snapshow(grph, canvas)

def test_texts_stressed_rotation():
    N = 300
    ANGLE_STEP=30
    canvas, grph = setup()
    locations = random_points(N)
    strings = double_shuffle(["Hello", "sympathy", "goodbye", "Aqua"], N)
    fonts = double_shuffle(["bold","italic","light","regular"], N)
    sizes = double_shuffle([18, 36, 72, 112],N)
    alpha = double_shuffle(["bug","mask","bug","mask"], N)
    rotations = double_shuffle(list(range(0,360,ANGLE_STEP)), N)
    grph.draw_strings(canvas, locations, strings, colors=random_colors(len(locations)), fonts=fonts, sizes=sizes, rotations=rotations, alphas=alpha)
    snapshow(grph, canvas)

def test_typesetting():
    canvas, grph = setup()
    lines = text_test_lines()
    x,y = 60,60
    locations = [(x,y+i*30) for i in range(len(lines)) if lines[i]]
    lines = [line for line in lines if line]
    sizes = [24] * len(lines)
    sizes[0] = 48
    colors = [None] * len(lines)
    colors[0] = [0.,1.,.1,0]
    for i in range(len(lines)):
        if "Tyger" in lines[i]:
            colors[i] = [1.0,.1,0,0.1]
    grph.draw_strings(canvas, locations, lines, sizes=sizes, colors=colors)
    snapshow(grph, canvas)

def test_one():
    canvas, grph = setup()
    grph.lines(canvas, [[0,0,ASIZE,ASIZE]], distance=1.5, color=RED, alpha=1.0)
    grph.circles(canvas, [[ASIZE/2,ASIZE/2, 20]],color=GREEN, alpha=1.0)
    grph.rects(canvas, [[ASIZE/4,ASIZE/4, ASIZE/4+40, ASIZE/4+40]],color=BLUE, alpha=1.0)
    grph.ellipses(canvas, [[ASIZE/4,ASIZE/2, 80, 40]],color=YELLOW, alpha=1.0)
    grph.triangles(canvas, [[ASIZE*3/4,ASIZE/2,40+ASIZE*3/4,20+ASIZE/2,40+ASIZE*3/4,-20+ASIZE/2 ]],color=WHITE, alpha=1.0)
    grph.draw_strings(canvas, [[ASIZE*3/4,ASIZE*3/4]], ["Hello world"], font="bold", color=PURPLE, alpha="mask", size=32)
    snapshow(grph, canvas)
