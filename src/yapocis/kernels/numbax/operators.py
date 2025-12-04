from __future__ import annotations

import numpy as np

try:
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - fallback
    HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def _wrap(f):
            return f

        return _wrap


@njit(cache=True, fastmath=True)
def _op_njit(a: np.ndarray, b: np.ndarray, out: np.ndarray, op: int) -> None:
    ar = a.ravel()
    br = b.ravel()
    orr = out.ravel()
    for i in range(orr.shape[0]):
        if op == 0:
            orr[i] = ar[i] + br[i]
        elif op == 1:
            orr[i] = ar[i] - br[i]
        elif op == 2:
            orr[i] = ar[i] * br[i]
        else:
            orr[i] = ar[i] / (br[i] + 1e-20)


def add(input1: np.ndarray, input2: np.ndarray, output: np.ndarray) -> np.ndarray:
    if HAVE_NUMBA:
        _op_njit(
            input1.astype(np.float32, copy=False),
            input2.astype(np.float32, copy=False),
            output,
            0,
        )
    else:
        np.add(input1, input2, out=output)
    return output


def add_res(input1: np.ndarray, input2: np.ndarray, output: np.ndarray) -> None:
    if HAVE_NUMBA:
        _op_njit(
            input1.astype(np.float32, copy=False),
            input2.astype(np.float32, copy=False),
            output,
            0,
        )
    else:
        np.add(input1, input2, out=output)


def sub(input1: np.ndarray, input2: np.ndarray, output: np.ndarray) -> np.ndarray:
    if HAVE_NUMBA:
        _op_njit(
            input1.astype(np.float32, copy=False),
            input2.astype(np.float32, copy=False),
            output,
            1,
        )
    else:
        np.subtract(input1, input2, out=output)
    return output


def sub_res(input1: np.ndarray, input2: np.ndarray, output: np.ndarray) -> None:
    if HAVE_NUMBA:
        _op_njit(
            input1.astype(np.float32, copy=False),
            input2.astype(np.float32, copy=False),
            output,
            1,
        )
    else:
        np.subtract(input1, input2, out=output)


def mul(input1: np.ndarray, input2: np.ndarray, output: np.ndarray) -> np.ndarray:
    if HAVE_NUMBA:
        _op_njit(
            input1.astype(np.float32, copy=False),
            input2.astype(np.float32, copy=False),
            output,
            2,
        )
    else:
        np.multiply(input1, input2, out=output)
    return output


def mul_res(input1: np.ndarray, input2: np.ndarray, output: np.ndarray) -> None:
    if HAVE_NUMBA:
        _op_njit(
            input1.astype(np.float32, copy=False),
            input2.astype(np.float32, copy=False),
            output,
            2,
        )
    else:
        np.multiply(input1, input2, out=output)


def div(input1: np.ndarray, input2: np.ndarray, output: np.ndarray) -> np.ndarray:
    if HAVE_NUMBA:
        _op_njit(
            input1.astype(np.float32, copy=False),
            input2.astype(np.float32, copy=False),
            output,
            3,
        )
    else:
        np.divide(input1, input2, out=output)
    return output


def div_res(input1: np.ndarray, input2: np.ndarray, output: np.ndarray) -> None:
    if HAVE_NUMBA:
        _op_njit(
            input1.astype(np.float32, copy=False),
            input2.astype(np.float32, copy=False),
            output,
            3,
        )
    else:
        np.divide(input1, input2, out=output)


# Alias divop -> div for consistency with interface
divop = div
divop_res = div_res
