from __future__ import annotations

from typing import Tuple, Any, Optional

import numpy as np


def infer_wh_from_args(args: Tuple[Any, ...]) -> Optional[Tuple[int, int]]:
    for a in args:
        if isinstance(a, np.ndarray):
            return a.shape
    return None
