DEBUG = False

#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.
import copy

from yapocis.lib.kerneldraw1 import KernelDraw1
from yapocis.lib.kerneldraw3 import KernelDraw3
from yapocis.utils.basic import *

BENCHMARKING = os.environ.get("BENCHMARKING", None)

from .basic import VIEWING_QUEUE


def sign(title: str, img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    width, height = img.shape[1], img.shape[0]
    if height > 480:
        fheight = height // 30
        fheight = 5 * (fheight // 5)
        if fheight % 5:
            fheight += 5
    else:
        fheight = 16
    if len(img.shape) == 2:
        ad = KernelDraw1(img)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        ad = KernelDraw3(img)
    ad.draw_string((10, 10), title, color=1.0, size=fheight, alpha="bug")
    ad.render()
    img = ad.read()
    return img


_showArrayCounter = 0


def sign_array(array: np.ndarray, *args, normalize=True):
    global _showArrayCounter
    _showArrayCounter += 1
    assert isinstance(array, np.ndarray)
    array = copy.deepcopy(array)
    title = " ".join([str(arg) for arg in args]).strip()
    if not title:
        title = "showarray %s" % _showArrayCounter

    if array.dtype != np.float32:
        array = array.astype(np.float32) / 255.0
    else:
        try:
            array = array[:, :]
        except:
            array = array[:, :, :]
    if normalize and array.dtype in (np.float32, np.float64):
        array -= array.min()
        array /= array.max()
    array = sign(title, array)
    return title, array


def show_array(array: np.ndarray, *args, **kwargs) -> None:
    if BENCHMARKING:
        return
    title, img = sign_array(array, *args, **kwargs)
    img = to_image(img)
    if img.mode == "L":
        img = img.convert("RGB")
    name = f"{title}.png"
    path = os.path.join(VIEWING_QUEUE, name)
    img.save(path)


def show_arraygrad(title, image, theta, grad=None) -> None:
    """
    Show an array with gradient orientation and optional magnitude
    Args:
        title: a useful title
        image: image to annotate
        theta: array of pi=1.0 angles
        grad: optional array of gradients, defaults to something visible

    Returns: None

    """
    image = copy.deepcopy(image)
    ad = KernelDraw3(image)
    # Decimate the arrays since we only use every 10th point any way
    theta = theta[::10, ::10]
    if grad is None:
        grad = np.zeros_like(theta)
        grad[:, :] = 20
    else:
        grad = grad[::10, ::10]
        grad -= grad.min()
        if grad.max() != 0.0:
            grad /= grad.max()
        grad *= 10
    cos = np.cos(theta * 3.14159)
    sin = np.sin(theta * 3.14159)
    dxs = grad * sin
    dys = grad * cos
    for i in range(10, image.shape[0], 10):
        for j in range(10, image.shape[1], 10):
            dx = dxs[i // 10, j // 10]
            dy = dys[i // 10, j // 10]
            grd = grad[i // 10, j // 10]
            x1, y1 = int(i - dx), int(j - dy)
            x2, y2 = int(i + dx), int(j + dy)
            color = grd / 10.0
            ad.line((x1, y1, x2, y2), distance=1.1, color=color, alpha=1)
    ad.render()
    a = ad.read()
    show_array(a, title, normalize=False)
