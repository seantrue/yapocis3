from __future__ import annotations

import numpy as np


def sum(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    np.add(a, b, out=c)
    return c


def sum_res(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    np.add(a, b, out=c)
    return c
