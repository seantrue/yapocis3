"""
Python typing
"""
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from typing import (Any, Callable, Dict, Iterable, Iterator, List, Optional,
                    Set, Tuple, Union)

import numpy as np
import PIL.Image

PILImage = PIL.Image.Image

Array = np.ndarray
Context = Dict[str, Any]
EngineID = Union[str, int]

Coords = np.ndarray
SimpleType = Union[str, int, float, bool, List, Tuple, Dict]
Config = Dict[str, SimpleType]

Image1 = np.ndarray
Image2 = np.ndarray
Image3 = np.ndarray
Image4 = np.ndarray
Image14 = Union[Image1, Image4]
Image1234 = Union[Image1, Image2, Image3, Image4]
