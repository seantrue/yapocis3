#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.
import os
import sys
from typing import List, Optional

from PIL import Image, ImageFont, ImageDraw

import yapocis
from yapocis.utils.typing import *

DEBUG = False


# bytescale, fromimage, toimage borrowed from scipy.misc
# Used under the liberal BSD license
# Returns a byte-scaled image
def byte_scale(data: Array, cmin: Optional[float] = None, cmax: Optional[float] = None, high: int = 255,
               low: int = 0) -> Array:
    if data.dtype == np.uint8:
        return data
    high = high - low
    cmin = data.min() if cmin is None else cmin
    cmax = data.max() if cmax is None else cmax
    scale = high * 1.0 / (cmax - cmin or 1)
    bytedata = ((data * 1.0 - cmin) * scale + 0.4999).astype(np.uint8)
    return bytedata + np.cast[np.uint8](low)


def from_image(im: PILImage, flatten: bool = False) -> Array:
    """
    Return a copy of a PIL image as a numpy array.

    Parameters
    ----------
    im : PIL image
        Input image.
    flatten : bool
        If true, convert the output to grey-scale.

    Returns
    -------
    fromimage : ndarray
        The different colour bands/channels are stored in the
        third dimension, such that a grey-image is MxN, an
        RGB-image MxNx3 and an RGBA-image MxNx4.

    """
    if not Image.isImageType(im):
        raise TypeError("Input is not a PIL image.")
    if flatten:
        im = im.convert('F')
    return np.array(im)


_errstr = "Mode is unknown or incompatible with input array shape."


def to_image(arr: Array, high: int = 255, low: int = 0, cmin: Optional[float] = None, cmax: Optional[float] = None,
             pal: Optional[Array] = None, mode: Optional[str] = None,
             channel_axis: Optional[int] = None, newaxis: Optional[int] = None) -> PILImage:
    """Takes a numpy array and returns a PIL image.  The mode of the
    PIL image depends on the array shape, the pal keyword, and the mode
    keyword.

    For 2-D arrays, if pal is a valid (N,3) byte-array giving the RGB values
    (from 0 to 255) then mode='P', otherwise mode='L', unless mode is given
    as 'F' or 'I' in which case a float and/or integer array is made

    For 3-D arrays, the channel_axis argument tells which dimension of the
      array holds the channel data.
    For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
      by default or 'YCbCr' if selected.
    if the

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
        if mode == 'F':
            data32 = data.astype(np.float32)
            image = Image.fromstring(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = byte_scale(data, high=high, low=low, cmin=cmin, cmax=cmax)
            image = Image.fromstring('L', shape, bytedata.tostring())
            if pal is not None:
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = np.arange(0, 256, 1, dtype=np.uint8)[:, newaxis] * \
                      np.ones((3,), dtype=np.uint8)[newaxis, :]
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.fromstring('1', shape, bytedata.tostring())
            return image
        if cmin is None:
            cmin = np.amin(np.ravel(data))
        if cmax is None:
            cmax = np.amax(np.ravel(data))
        data = (data * 1.0 - cmin) * (high - low) / (cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(np.uint32)
            image = Image.fromstring(mode, shape, data32.tostring())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
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
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = np.transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = np.transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.fromstring(mode, shape, strdata)
    return image


def newsize(img: PILImage, width: int, height: int, longest: int) -> Tuple[bool, int, int]:
    if longest is not None:
        width, height = img.size[0], img.size[1]
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


def imread(filename: str, longest: Optional[int] = None, keep_alpha:bool=False) -> Union[Image1,Image3]:
    img = Image.open(filename)
    mode = img.mode
    if mode == "P":
        img = img.convert("RGBA")
    width, height = img.size
    resize,w,h = newsize(img, width, height, longest)
    if resize:
        img = img.resize((w, h))
    a = from_image(img).astype(np.float32)
    if not keep_alpha and len(a.shape) == 3 and a.shape[2] == 4:
        a = a[:, :, 0:3]
    return a


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
    try:
        font = ImageFont.truetype("zapfino.ttf", fheight)
    except:
        font = ImageFont.truetype("/Library/Fonts/AppleGothic.ttf", fheight)
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


def _show_array(*args) -> Tuple[str, PILImage]:
    global _showArrayCounter
    _showArrayCounter += 1
    array = args[-1]
    args = args[:-1]
    title = " ".join([str(arg) for arg in args]).strip()
    if not title:
        title = f"Array {_showArrayCounter}"
    if len(array.shape) == 3 and array.shape[2] == 4:
        array = array[:,:,:3]
    if array.dtype != np.uint8:
        array = array.copy()
        if array.dtype in (np.float32, np.float64, float):
            array -= array.min()
            array /= array.max()
            array *= 255.
            array = array.astype(np.uint8)
    img = Image.fromarray(array)
    shape = array.shape
    font = ImageFont.truetype("/Library/Fonts/AppleGothic.ttf", 25)
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
    if len(shape) > 2:
        ink = [ink] * shape[-1]
        ink = tuple(ink)
    draw.text((10, 10), title, ink, font=font)
    return title, img


def show_array(*args) -> None:
    title, img = _show_array(*args)
    img.show()


def show_arraygrad(title: str, image: Image1, theta: Image1, grad: Optional[Image1] = None) -> None:
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
    title, img = _show_array(title, image)
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
    img.show()


class Shaper:
    def __init__(self, data:Array):
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
        if self.order == 'C':
            return self.data
        if self.order:
            self.data = self.asimage()
        if self.order == None:
            self.order = 'C'
            self.data = self.data.reshape(-1, order=self.order)
            return self.data

    def ascols(self):
        if self.order == 'F':
            return self.data
        if self.order:
            self.data = self.asimage()
        if self.order == None:
            self.order = 'F'
            self.data = self.data.reshape(-1, order=self.order)
        return self.data


def histeq_large(im: Image1, nbr_bins=2 ** 16) -> Array:
    """
    Histogram equalization of a 2D array

    Args:
        im: 2D Array
        nbr_bins: number of bins

    Returns:
        2d Array, dtype=np.float32
    """
    imhist, bins = np.histogram(im.flatten(), nbr_bins, normed=True)
    imhist[0] = 0

    cdf = imhist.cumsum()  # cumulative distribution function
    # TODO: And here we have a non-linear something or other
    # cdf ** .5
    # TODO: This normalization looks wrong for a different number of bins.
    cdf = (nbr_bins- 1) * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(), bins[:-1], cdf)
    return np.array(im2, np.float32).reshape(im.shape)


def histeq(im: Image1, nbr_bins: int = 256) -> Tuple[Array, Array]:
    """
    Histogram equalization quantized to 8 bits. Appropriate for 8 bit images.
    Args:
        im: 2d array of pixels
        nbr_bins: defaults to 256

    Returns: array like im, and the cumulative density table

    """
    imhist, bins = np.histogram(im.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize
    im2 = np.interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape), cdf


ALIGNMENT = 4


def align_image(image: Image1, alignment: int = ALIGNMENT) -> Image1:
    """
    OpenCL really seems to prefer data that is aligned on quadfloat boundaries.
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
    Ensures that array contents are in range 0 <= elem < 1.0
    Args:
        a: an array to be normalized

    Returns: an array whose min and max values are 0.0 and really close to 1.0

    """
    a = a - a.min()
    a = a / (a.max() * (1.0-np.epsilon))
    return a


def set_margin(a: Image1, margin: int, value: Optional[float] = 0.0) -> None:
    """
    Set the edges of a 2D array to a constant value
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


def set_frame(a: Image1, width: int, frame: Optional[float] = 1.0, margin: Optional[float] = 0.0) -> None:
    """
    Set the margin and then set the outer most pixels.

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

def chunker(data: Iterable, chunk_size:int=100) -> Iterator[List]:
    """
    Chop an iterable into a digestable chunks
    Args:
        data: an iterable
        chunk_size: defaults to 100

    Returns: a generator of lists

    """
    for i in range(0, len(data), chunk_size):
        yield data[i:i+chunk_size]


def get_project_dir() -> Optional[str]:
    yapocis_project = os.environ.get("YAPOCIS_PROJECT",None)
    if yapocis_project and os.path.isdir(yapocis_project):
        return yapocis_project
    else:
        return None

def get_yapocis_dirs() -> List[str]:
    yapocis_dir = os.path.dirname(yapocis.__file__)
    current_dir = os.getcwd()
    directories = [yapocis_dir, current_dir]
    project_dir = get_project_dir()
    if project_dir:
        directories.append(project_dir)
    return directories

def find_file(dirs: List[str], subdirs: List[str], filename: str) -> Optional[str]:
    for d in dirs:
        for sd in subdirs:
            pth = os.path.join(d, sd, filename)
            if os.path.exists(pth):
                return pth
    return None


def find_yapocis_file(subdirs:List[str], filename:str) -> Optional[str]:
    dirs = get_yapocis_dirs()
    tfilename = find_file(dirs, subdirs, filename)
    return tfilename
