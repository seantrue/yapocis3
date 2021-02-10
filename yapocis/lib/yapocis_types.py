"""
Python typing
"""
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import numpy as np
from typing import Dict, Any, List, Union, Optional, Callable, Set, Tuple
import PIL.Image
Context = Dict[str, Any]
Array = np.ndarray
EngineID = Union[str, int]
PILImage = PIL.Image.Image