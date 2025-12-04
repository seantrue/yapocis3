"""
Python typing
"""

#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

from typing import Any, Union

import numpy as np
import numpy.typing as npt
import PIL.Image

# Re-export for backward compatibility
dict = dict
list = list
tuple = tuple
set = set

PILImage = PIL.Image.Image

Array = npt.NDArray
Context = dict[str, Any]
DeviceID = Union[str, int]

Coords = npt.NDArray
SimpleType = Union[str, int, float, bool, list, tuple, dict]
Config = dict[str, SimpleType]
PixelType = Union[np.uint8, np.float32]

Image1 = npt.NDArray[PixelType]
Image2 = npt.NDArray[PixelType]
Image3 = npt.NDArray[PixelType]
Image4 = npt.NDArray[PixelType]
Image14 = Union[Image1, Image4]
Image1234 = Union[Image1, Image2, Image3, Image4]
