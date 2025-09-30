from __future__ import annotations

import numpy as np


def _zcs_generic(
    width: int,
    height: int,
    threshold: float,
    input: np.ndarray,
    zcs: np.ndarray,
    planes: int,
):
    # input expected flattened with last dim planes or 2D with planes=1
    if planes == 1 and input.ndim == 2:
        img = input
    else:
        h, w = height, width
        img = input.reshape(h, w, planes)
    if planes == 1:
        # compute gradient-like delta and sign changes in 4-neighborhood
        ap = np.pad(img, ((1, 1), (1, 1)), mode="edge")
        above = ap[2:, 1:-1]
        below = ap[:-2, 1:-1]
        left = ap[1:-1, :-2]
        right = ap[1:-1, 2:]
        delta = (above - below) ** 2 + (right - left) ** 2
        signs = lambda a: np.sign(a)
        zmask = (delta > threshold) & (
            (signs(above) != signs(below)) | (signs(left) != signs(right))
        )
        zcs[:] = zmask.astype(np.float32)
    else:
        ap = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
        above = ap[2:, 1:-1, :]
        below = ap[:-2, 1:-1, :]
        left = ap[1:-1, :-2, :]
        right = ap[1:-1, 2:, :]
        delta = (above - below) ** 2 + (right - left) ** 2
        zmask = (delta > threshold) & (
            (np.sign(above) != np.sign(below)) | (np.sign(left) != np.sign(right))
        )
        zcs[:] = zmask.astype(np.float32)
    return zcs


def zcs1(width: int, height: int, threshold: float, input: np.ndarray, zcs: np.ndarray):
    return _zcs_generic(width, height, threshold, input, zcs, 1)


def zcs2(width: int, height: int, threshold: float, input: np.ndarray, zcs: np.ndarray):
    return _zcs_generic(width, height, threshold, input, zcs, 2)


def zcs3(width: int, height: int, threshold: float, input: np.ndarray, zcs: np.ndarray):
    return _zcs_generic(width, height, threshold, input, zcs, 3)


def zcs4(width: int, height: int, threshold: float, input: np.ndarray, zcs: np.ndarray):
    return _zcs_generic(width, height, threshold, input, zcs, 4)


def sub(input1: np.ndarray, input2: np.ndarray, output: np.ndarray):
    np.subtract(input1, input2, out=output)
    return output


def sub_res(input1: np.ndarray, input2: np.ndarray, output: np.ndarray):
    np.subtract(input1, input2, out=output)
