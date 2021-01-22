
import numpy as np

from yapocis.rpc import kernels

draw_cl = rects_cl = circles_cl = None
kernels.import_program("graphics", draw="draw_cl", rects="rects_cl", circles="circles_cl")


def draw(image, coords, distance=1.0):
    if not hasattr(coords, "dtype"):
        coords = np.array(coords, dtype=np.float32)
    assert len(coords.shape) == 2 and coords.shape[1] == 4
    draw_cl(image, coords, distance)

def rects(image, coords, rcolor=1.0):
    if not hasattr(coords, "dtype"):
        coords = np.array(coords, dtype=np.float32)
    assert len(coords.shape) == 2 and coords.shape[1] == 4
    rects_cl(image, coords, rcolor)

def circles(image, coords, rcolor=1.0):
    if not hasattr(coords, "dtype"):
        coords = np.array(coords, dtype=np.float32)
    assert len(coords.shape) == 2 and coords.shape[1] == 4
    circles_cl(image, coords, rcolor)
