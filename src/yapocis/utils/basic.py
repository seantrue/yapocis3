#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.
import os
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

import skimage
from PIL import Image, ImageDraw, ImageFont

from yapocis.fonts import font as find_font
from yapocis.utils.stereo_typing import *

from .internals import get_viewing_queue

VIEWER = os.environ.get("YAPOCIS_VIEWER")
VIEWING_QUEUE = get_viewing_queue()
RAW_QUEUE = os.path.join(VIEWING_QUEUE, "raw")
os.makedirs(RAW_QUEUE, exist_ok=True)

if not os.path.isdir(VIEWING_QUEUE):
    raise RuntimeError(f"Can't find viewing queue '{VIEWING_QUEUE}'")

DEBUG = False
BENCHMARKING = os.environ.get("BENCHMARKING", None)


def get_test_image() -> "Image3":
    """Load the standard test image for documentation examples.

    Returns:
        A float32 RGB image array normalized to [0, 1].
    """
    from yapocis.images import find_image

    return imread(find_image("test"))


def _current_backend() -> str:
    """Return the active backend name for labeling output images.

    Falls back to 'ocl' if not set. Keeps value simple and lowercase.
    """
    b = os.environ.get("YAPOCIS_BACKEND")
    return (b or "ocl").lower()


# Image conversions run over slabs of this many rows, so float32 temporaries stay
# small instead of being full-size copies (numpy does float16 math in software).
_SLAB_ROWS = 64


def _row_slabs(a: Array) -> Iterator[tuple[slice, Array]]:
    for r in range(0, a.shape[0], _SLAB_ROWS):
        rows = slice(r, r + _SLAB_ROWS)
        yield rows, a[rows]


def _to_unit_float(img: Array, max_value: float, dtype: np.dtype) -> Array:
    """img / max_value as dtype, computed in float32 (float64 for wider dtypes) one slab at a time."""
    work = np.float64 if np.dtype(dtype).itemsize > 4 else np.float32
    out = np.empty(img.shape, dtype=dtype)
    for rows, slab in _row_slabs(img):
        out[rows] = slab.astype(work) / work(max_value)
    return out


def _byte_scale_float16(data: Array, cmin, cmax, high: int, low: int) -> Array:
    """byte_scale for float16: the same float32 math, one slab at a time."""
    if cmin is None or cmax is None:
        lo, hi = np.float32(np.inf), np.float32(-np.inf)
        for _, slab in _row_slabs(data):
            slab = slab.astype(np.float32)
            lo, hi = np.minimum(lo, slab.min()), np.maximum(hi, slab.max())
        cmin = lo if cmin is None else cmin
        cmax = hi if cmax is None else cmax
    out = np.empty(data.shape, dtype=np.uint8)
    for rows, slab in _row_slabs(data):
        out[rows] = byte_scale(slab.astype(np.float32), cmin, cmax, high, low)
    return out


# bytescale, toimage borrowed from scipy.misc
# Used under the liberal BSD license
# Returns a byte-scaled image
def byte_scale(
    data: Array,
    cmin: float | None = None,
    cmax: float | None = None,
    high: int = 255,
    low: int = 0,
) -> Array:
    if data.dtype == np.uint8:
        return data
    if data.dtype == np.float16 and data.ndim:
        return _byte_scale_float16(data, cmin, cmax, high, low)
    high = high - low
    cmin = data.min() if cmin is None else cmin
    cmax = data.max() if cmax is None else cmax
    scale = high * 1.0 / (cmax - cmin or 1)
    bytedata = ((data * 1.0 - cmin) * scale + 0.4999).astype(np.uint8)
    return bytedata + np.uint8(low)


_errstr = "Mode is unknown or incompatible with input array shape."


def to_image(
    arr: Array,
    high: int = 255,
    low: int = 0,
    cmin: float | None = None,
    cmax: float | None = None,
    pal: Array | None = None,
    mode: str | None = None,
    channel_axis: int | None = None,
    newaxis: int | None = None,
) -> PILImage:
    """Takes a numpy array and returns a PIL image.  The mode of the
    PIL image depends on the array shape, the pal keyword, and the mode
    keyword.

    For 2-D arrays, if pal is a valid (N,3) byte-array giving the RGB values
    (from 0 to 255) then mode='P', otherwise mode='L', unless mode is given
    as 'F' or 'I' in which case a float and/or integer array is made

    For 3-D arrays, the channel_axis argument tells which dimension of the
    array holds the channel data. If one of the dimensions is 3, the mode is
    'RGB' by default or 'YCbCr' if selected.

    The numpy array must be either 2 dimensional or 3 dimensional.
    """
    data = np.asarray(arr)
    if np.iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == "F":
            data32 = data.astype(np.float32)
            image = Image.frombytes(mode, shape, data32.tobytes())
            return image
        if mode in [None, "L", "P"]:
            bytedata = byte_scale(data, high=high, low=low, cmin=cmin, cmax=cmax)
            image = Image.frombytes("L", shape, bytedata.tobytes())
            if pal is not None:
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == "P":  # default gray-scale
                pal = (
                    np.arange(0, 256, 1, dtype=np.uint8)[:, newaxis]
                    * np.ones((3,), dtype=np.uint8)[newaxis, :]
                )
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
            return image
        if mode == "1":  # high input gives threshold for 1
            bytedata = data > high
            image = Image.frombytes("1", shape, bytedata.tobytes())
            return image
        if cmin is None:
            cmin = np.amin(np.ravel(data))
        if cmax is None:
            cmax = np.amax(np.ravel(data))
        data = (data * 1.0 - cmin) * (high - low) / (cmax - cmin) + low
        if mode == "I":
            data32 = data.astype(np.uint32)
            image = Image.frombytes(mode, shape, data32.tobytes())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if 3 in shape:
            ca = np.flatnonzero(np.asarray(shape) == 3)[0]
        else:
            ca = np.flatnonzero(np.asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    strdata = None
    bytedata = byte_scale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tobytes()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = np.transpose(bytedata, (0, 2, 1)).tobytes()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = np.transpose(bytedata, (1, 2, 0)).tobytes()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = "RGB"
        else:
            mode = "RGBA"

    if mode not in ["RGB", "RGBA", "YCbCr", "CMYK"]:
        raise ValueError(_errstr)

    if mode in ["RGB", "YCbCr"]:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ["RGBA", "CMYK"]:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image


def newsize(width: int, height: int, longest: int) -> tuple[bool, int, int]:
    if longest is not None:
        resize = False
        if width > height and width > longest:
            height = int(float(height / width) * longest)
            width = longest
            resize = True
        elif height >= width and height > longest:
            width = int(float(width / height) * longest)
            height = longest
            resize = True
        return resize, width, height
    return False, width, height


def imread(
    filename: str,
    longest: int | None = None,
    keep_alpha: bool = False,
    dtype: npt.DTypeLike = np.float32,
) -> Union[Image1, Image3]:
    """Read an image as floats in [0, 1].

    dtype=np.float16 converts straight to half precision without a full-size
    float32 copy; values are identical to imread(...).astype(np.float16).
    """
    img = skimage.io.imread(filename)
    if keep_alpha:
        # skimage drops tRNS transparency (palette/grayscale PNGs) when
        # converting to RGB; let PIL expand it into a real alpha channel.
        with Image.open(filename) as pimg:
            if pimg.mode in ("P", "PA", "LA", "La") or "transparency" in pimg.info:
                img = np.asarray(pimg.convert("RGBA"))
    img_dtype = img.dtype
    resize, w, h = newsize(img.shape[1], img.shape[0], longest)
    if resize:
        img = skimage.transform.resize(
            img, (h, w), preserve_range=True, anti_aliasing=True
        )
    if not keep_alpha and len(img.shape) == 3 and img.shape[2] == 4:
        img = img[:, :, 0:3]
    return _to_unit_float(img, np.iinfo(img_dtype).max, dtype)


def imwrite(filename: str, img: Union[Image1, Image3, Image4]):
    """Save a [0, 1] float image. Out-of-range values are clipped, not rescaled."""
    img = np.clip(np.nan_to_num(img, nan=0.0, neginf=0.0, posinf=1.0), 0.0, 1.0)
    to_image(img, cmin=0.0, cmax=1.0).save(filename)


import time


class Stage:
    def __init__(self):
        self.t = None
        self.stage = None

    def __call__(self, *args):
        stage = " ".join([str(arg) for arg in args])
        if self.stage:
            t = time.time()
            print(self.stage, "done in", t - self.t)
        if args:
            print("Start", stage)
            self.t = time.time()
            self.stage = stage
        else:
            self.t = self.stage = None


stage = Stage()


def sign(title: str, img: Union[Image1, PILImage]) -> PILImage:
    if hasattr(img, "dtype"):
        img = to_image(img)
    shape = img.size
    width, height = shape[0], shape[1]
    if height > 480:
        fheight = height / 30
        fheight = 5 * (fheight / 5)
        if fheight % 5:
            fheight += 5
    else:
        fheight = 16
    font = ImageFont.truetype(find_font("default"), size=fheight)
    draw = ImageDraw.Draw(img)
    try:
        pixels = [img.getpixel((x, height - 15)) for x in range(10, 30)]
        pixels = np.array(pixels)
        darkness = pixels.sum() / pixels.size
        if darkness < 128:
            ink = 255
        else:
            ink = 0
    except:
        ink = 0
    if len(shape) > 2:
        ink = tuple(ink, ink, ink)
    if ink:
        ink = "white"
    else:
        ink = "black"
    draw.draw_string((10, height - 75), title, ink, font=font)
    return img


_showArrayCounter = 0


def array_to_image(array):
    if len(array.shape) == 3 and array.shape[2] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        # float16 is widened: numpy does half-precision math in software
        array = array.astype(np.float32) if array.dtype == np.float16 else array.copy()
        if array.dtype in (np.float32, np.float64, float):
            array -= array.min()
            mx = array.max()
            array /= mx if mx != 0.0 else 1.0
            array *= 255.0
            array = array.astype(np.uint8)
    img = Image.fromarray(array)
    if img.mode == "L":
        img = img.convert("RGB")
    return img


def sign_array(*args, **kwargs) -> tuple[str, PILImage]:
    global _showArrayCounter
    _showArrayCounter += 1
    array = args[-1]
    args = args[:-1]
    title = " ".join([str(arg) for arg in args]).strip()
    # Append backend tag to the title/filename for clarity in multi-backend runs
    backend = _current_backend()
    if backend:
        title = f"{title} [{backend}]" if title else f"[{backend}]"
    if not title:
        title = f"Array {_showArrayCounter}"
    img = array_to_image(array)
    shape = array.shape
    font = ImageFont.truetype(find_font("default"), size=22)
    draw = ImageDraw.Draw(img)
    try:
        pixels = [img.getpixel((x, 15)) for x in range(10, 30)]
        pixels = np.array(pixels)
        darkness = pixels.sum() / pixels.size
        if darkness < 128:
            ink = 255
        else:
            ink = 0
    except IndexError:
        ink = 0
    caption = (
        ""
        if not kwargs
        else " " + " ".join([f"{key}:{value}" for (key, value) in kwargs.items()])
    )
    if len(shape) > 2:
        ink = [ink] * shape[-1]
        ink = tuple(ink)
    draw.text((10, 10), title + caption, ink, font=font)
    return title, img


def show_array(array: np.ndarray, *args, queue_dir=None, checkzero=True, **kwargs) -> Path:
    queue_dir = queue_dir or VIEWING_QUEUE
    raw_dir = os.path.join(queue_dir, "raw") if queue_dir else RAW_QUEUE
    split_args = []
    for arg in args:
        arg = str(arg)
        split_args.extend(arg.split())
    final_args = [split_args[0]] if split_args else []
    for arg in split_args[1:]:
        if arg not in final_args:
            final_args.append(arg)
    args = tuple(final_args)
    if BENCHMARKING:
        print(args, file=sys.stderr)
        return
    if checkzero:
        assert array.flatten().sum() > 0, f"No one likes an empty picture {args}"
    raw_img = array_to_image(array)
    title, img = sign_array(*args, array, **kwargs)
    name = f"{title}.png"
    pth = os.path.join(queue_dir, name)
    img.save(pth)
    if os.path.exists(raw_dir):
        raw_img.save(os.path.join(raw_dir, name))
        npz_name = f"{title}.npz"
        np.savez_compressed(os.path.join(raw_dir, npz_name), array)
    if VIEWER:
        os.system(f'{VIEWER} "{os.path.abspath(pth)}"&')
    return Path(pth)

def show_arraygrad(
    title: str, image: Image1, theta: Image1, grad: Image1 | None = None
) -> None:
    """
    Show an array with gradient orientation and optional magnitude
    Args:
        title: a useful title
        image: image to annotate
        theta: array of pi=1.0 angles
        grad: optional array of gradients, defaults to something visible

    Returns: None

    """
    from PIL import ImageDraw  # @UnresolvedImport

    title, img = sign_array(title, image)
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size
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
    for x in range(10, width - 10, 5):
        for y in range(10, height - 10, 5):
            i, j = y, x
            dx = grad[i, j] * sin[i, j]
            dy = grad[i, j] * cos[i, j]
            try:
                x1, y1 = int(x - dx), int(y - dy)
                x2, y2 = int(x + dx), int(y + dy)
            except ValueError:
                # NANs happen
                continue
            a, b, c = img.getpixel((x, y))
            grey = (a + b + c) // 3
            if grey > 128:
                color = "black"
            else:
                color = "white"
            draw.line([(x1, y1), (x2, y2)], fill=color)
    return show_array(np.array(img), title)


class Shaper:
    def __init__(self, data: Array):
        assert len(data.shape) == 2, "Shaper requires 2-d data"
        self.shape = data.shape
        self.order = None
        self.data = data[:, :]

    def update(self, data):
        assert self.data.shape == data.shape, "Must conform"
        self.data = data

    def asimage(self):
        if self.order:
            self.data = self.data.reshape(self.shape, order=self.order)
            self.order = None
        return self.data

    def asrows(self):
        if self.order == "C":
            return self.data
        if self.order:
            self.data = self.asimage()
        if self.order == None:
            self.order = "C"
            self.data = self.data.reshape(-1, order=self.order)
            return self.data

    def ascols(self):
        if self.order == "F":
            return self.data
        if self.order:
            self.data = self.asimage()
        if self.order == None:
            self.order = "F"
            self.data = self.data.reshape(-1, order=self.order)
        return self.data


def histeq_large(im: Image1, nbr_bins=2**16) -> Array:
    """
    Histogram equalization of a 2D array

    Args:
        im: 2D Array
        nbr_bins: number of bins

    Returns:
        2d Array, dtype=np.float32
    """
    imhist, bins = np.histogram(im.ravel(), nbr_bins, density=True)
    imhist[0] = 0

    cdf = imhist.cumsum()  # cumulative distribution function
    # TODO: And here we have a non-linear something or other
    # cdf ** .5
    # TODO: This normalization looks wrong for a different number of bins.
    cdf = (nbr_bins - 1) * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.ravel(), bins[:-1], cdf)
    return np.array(im2, np.float32).reshape(im.shape)


def histeq(im: Image1, nbr_bins: int = 256, max_value=255) -> tuple[Array, Array]:
    """
    Histogram equalization quantized to 8 bits. Appropriate for 8 bit images.
    Args:
        im: 2d array of pixels
        nbr_bins: defaults to 256

    Returns: array like im, and the cumulative density table

    """
    imhist, bins = np.histogram(im.ravel(), nbr_bins, normed=True)
    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = max_value * cdf / cdf[-1]  # normalize
    im2 = np.interp(im.ravel(), bins[:-1], cdf)
    return im2.reshape(im.shape), cdf


ALIGNMENT = 4


def align_image(image: Image1, alignment: int = ALIGNMENT) -> Image1:
    """
    Some drivers really seems to prefer data that is aligned on quadfloat boundaries.
    Args:
        image: 2d or 3d arrau
        alignment: defaults to 4

    Returns: an array whose dimensions are a multiple of alignment.

    """
    assert len(image.shape) in (2, 3)
    width, height = image.shape[:2]
    oddw, oddh = width % alignment, height % alignment
    if oddw == 0 and oddh == 0:
        return image.copy()
    evenw = width - oddw if oddw else width
    evenh = height - oddh if oddh else height
    if len(image.shape) == 2:
        aligned = np.empty((evenw, evenh), dtype=image.dtype)
        aligned[:, :] = image[:evenw, :evenh]
    else:
        aligned = np.empty((evenw, evenh, image.shape[2]), dtype=image.dtype)
        aligned[:, :, :] = image[:evenw, :evenh, :]
    return aligned


def normalize(a: Array) -> Array:
    """
    Ensures that array contents are in range 0 <= elem <= 1.0
    Args:
        a: an array to be normalized

    Returns: an array whose min and max values are 0.0 and really close to 1.0

    """
    a = a - a.min()
    a = a / a.max()
    return a


def set_margin(a: Image1, margin: int, value: float | None = 0.0) -> None:
    """
    set the edges of a 2D array to a constant value
    Args:
        a: a 2D array
        margin: width of margin
        value: value to set, defaults to 0.0

    Returns: None

    """
    a[:, -margin:] = value
    a[:, :margin] = value
    a[-margin:, :] = value
    a[:margin, :] = value


def set_frame(
    a: Image1, width: int, frame: float | None = 1.0, margin: float | None = 0.0
) -> None:
    """
    set the margin and then set the outer most pixels.

    Args:
        a: 2D Image
        width: width of clearm
        frame: edge pixel defailts to 1
        margin: margin pixels default to 0

    Returns:

    """
    set_margin(a, width, margin)
    a[0, :] = frame
    a[:, 0] = frame
    a[-1, :] = frame
    a[:, -1] = frame


def compute_360_to_180(rotation: float) -> float:
    rotation = rotation % 360
    if rotation > 180:
        rotation = -180 + (rotation - 180)
    return rotation


def chunker(data: Iterable, chunk_size: int = 100) -> Iterator[list]:
    """
    Chop an iterable into a digestable chunks
    Args:
        data: an iterable
        chunk_size: defaults to 100

    Returns: a generator of lists

    """
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


def get_image_depth(image):
    if len(image.shape) in (2, 3):
        if len(image.shape) == 2:
            return 1
        elif image.shape[2] == 1:
            return 1
        elif image.shape[2] == 2:
            return 2
        elif image.shape[2] == 3:
            return 3
        elif image.shape[2] == 4:
            return 4
    raise ValueError(f"Array shape {image.shape} is not supported")
