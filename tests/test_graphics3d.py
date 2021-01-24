
import numpy as np
from yapocis.utils import imread
from yapocis.utils import show_array, show_arraygrad
from yapocis.kernels.median import median3x3
from yapocis.kernels.gradient import gradient
from yapocis.kernels.hsi import rgb2hsi, hsi2rgb
from yapocis.channels import split_channels, join_channels

from yapocis.kernels.graphics import ArrayDraw3D

def show_arraygrad(title, image, theta, grad = None) -> None:
    """
    Show an array with gradient orientation and optional magnitude
    Args:
        title: a useful title
        image: image to annotate
        theta: array of pi=1.0 angles
        grad: optional array of gradients, defaults to something visible

    Returns: None

    """
    ad = ArrayDraw3D(image)
    height, width = theta.shape
    if grad is None:
        grad = np.zeros_like(theta)
        grad[:, :] = 5
    else:
        grad = grad.copy()
        grad -= grad.min()
        if grad.max() != 0.0:
            grad /= grad.max()
        grad *= 10
    cos = np.cos(theta * 3.14159)
    sin = np.sin(theta * 3.14159)
    lines = []
    colors = []
    dxs = grad * sin
    dys = grad * cos
    for i in range(10, image.shape[0], 10):
        for j in range(10, image.shape[1], 10):
            dx = dxs[i,j]
            dy = dys[i,j]
            x1, y1 = int(i - dx), int(j - dy)
            x2, y2 = int(i + dx), int(j + dy)
            grey = image[i,j,:].sum()/3.0
            if grey > .5:
                color = (0,0,0)
            else:
                color = (1,1,1)
            lines.append((x1,y1,x2,y2))
            colors.append(color)
    ad.lines(lines,distance=1.5, colors=colors,alpha=.75)
    ad.render()
    show_array("Gradient", image)

def test_graphics3d():
    image = np.float32(imread("tests/test.jpg"))
    image /= 256.

    r, g, b = split_channels(image)
    h, s, i = rgb2hsi(r, g, b)
    g, a = gradient(i, 5)
    show_array("Gradient", g)
    show_arraygrad("Grad angle", image, a)
    show_arraygrad("Grad vectors", image, a, g * 10)