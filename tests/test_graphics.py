import numpy as np
from random import randint

from yapocis.utils import show_array
from yapocis.channels import join_channels
from yapocis.kernels.graphics import ArrayDraw, ArrayDraw3D
from yapocis.kernels.graphics import lines, rects, circles, triangles, ellipses

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
    a, ellipses = random_points(n=100)
    a = np.zeros((1000, 1000)).astype(np.float32)
    radius = 50
    for ellipse in ellipses:
        ellipse += [randint(2, radius), randint(2, radius)]
    return a, ellipses


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

def test_arraydraw():
    a, random_coords = random_ellipses()
    a[:] = 1.0
    ad = ArrayDraw(a)
    _, circles = random_circles()
    _, rects = random_lines()
    _, lines = random_lines()
    ad.ellipses(random_coords, random_colors(), random_colors())
    ad.circles(circles, random_colors(), random_colors())
    ad.rects(rects, random_colors(), random_colors())
    ad.lines(lines, 1.5, random_colors(), random_colors())
    ad.render()
    show_array(f"arraydraw", a)

def test_arraydraw3d():
    a, random_coords = random_ellipses()
    a = join_channels(a,a,a)
    ad = ArrayDraw3D(a)
    _, circles = random_circles()
    _, rects = random_lines()
    _, lines = random_lines()
    _, triangles = random_triangles()
    _, ellipses = random_ellipses()
    alpha = random_colors()
    ad.set_color((1,0,0))
    ad.ellipses(random_coords, alpha=alpha)
    ad.set_color((0,1,0))
    ad.circles(circles, alpha=alpha)
    ad.set_color((0,0,1))
    ad.rects(rects, alpha=alpha)
    ad.set_color((1,1,0))
    ad.lines(lines, distance=1, alpha=alpha)
    ad.set_color((0,1,1))
    ad.triangles(triangles, colors=(random_colors(),random_colors(),random_colors()),alpha=alpha)
    ad.set_color((1,1,1))
    ad.ellipses(ellipses, (.5,.5,.2), alpha=alpha)
    ad.render()
    show_array(f"arraydraw3d", a)



