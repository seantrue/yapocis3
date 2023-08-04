"""
Python typing
"""
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

from typing import (Any, Callable, Dict, Iterable, Iterator, List, Optional,
                    Set, Tuple, Union)

import numpy as np
import numpy.typing as npt
import PIL.Image
PILImage = PIL.Image.Image

Array = npt.NDArray
Context = Dict[str, Any]
EngineID = Union[str, int]

Coords = npt.NDArray
SimpleType = Union[str, int, float, bool, List, Tuple, Dict]
Config = Dict[str, SimpleType]
PixelType = Union[np.uint8,np.float32]


Image1 = npt.NDArray[PixelType]
Image2 = npt.NDArray[PixelType]
Image3 = npt.NDArray[PixelType]
Image4 = npt.NDArray[PixelType]
Image14 = Union[Image1, Image4]
Image1234 = Union[Image1, Image2, Image3, Image4]
