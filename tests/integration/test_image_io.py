"""imread/imwrite and the float -> uint8 paths, including float16."""

import numpy as np
import pytest
import skimage.io
import skimage.transform

from yapocis.images import find_image
from yapocis.utils.basic import (
    array_to_image,
    byte_scale,
    imread,
    imwrite,
    newsize,
    to_image,
)


def _reference_imread(filename, longest=None):
    """imread as it was before the dtype option: full float32 conversion."""
    img = skimage.io.imread(filename)
    img_dtype = img.dtype
    resize, w, h = newsize(img.shape[1], img.shape[0], longest)
    if resize:
        img = skimage.transform.resize(
            img, (h, w), preserve_range=True, anti_aliasing=True
        )
    return img.astype(np.float32) / np.iinfo(img_dtype).max


@pytest.mark.parametrize("longest", [None, 300])
def test_imread_float32_unchanged(longest):
    path = find_image("test")
    a = imread(path, longest=longest)
    ref = _reference_imread(path, longest)
    assert a.dtype == np.float32
    assert a.shape == ref.shape
    assert np.array_equal(a, ref)


@pytest.mark.parametrize("longest", [None, 300])
def test_imread_float16_matches_float32_route(longest):
    path = find_image("test")
    a16 = imread(path, longest=longest, dtype=np.float16)
    ref = imread(path, longest=longest).astype(np.float16)
    assert a16.dtype == np.float16
    assert np.array_equal(a16.view(np.uint16), ref.view(np.uint16))


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_imread_alpha(tmp_path, dtype):
    rgba = np.random.default_rng(4).integers(0, 256, (30, 40, 4), dtype=np.uint8)
    path = tmp_path / "rgba.png"
    skimage.io.imsave(path, rgba, check_contrast=False)
    rgb = imread(str(path), dtype=dtype)
    assert rgb.dtype == dtype and rgb.shape == (30, 40, 3)
    assert np.array_equal(rgb, (rgba[:, :, :3] / np.float32(255)).astype(dtype))
    with_alpha = imread(str(path), keep_alpha=True, dtype=dtype)
    assert with_alpha.shape == (30, 40, 4)
    assert np.array_equal(with_alpha, (rgba / np.float32(255)).astype(dtype))


@pytest.mark.parametrize(
    "kwargs", [{}, {"cmin": 0.0, "cmax": 1.0}, {"cmin": 0.25}, {"high": 200, "low": 10}]
)
def test_byte_scale_float16_matches_float32(kwargs):
    rng = np.random.default_rng(0)
    a32 = rng.uniform(0.1, 0.9, (300, 211, 3)).astype(np.float16).astype(np.float32)
    a16 = a32.astype(np.float16)
    assert np.array_equal(byte_scale(a16, **kwargs), byte_scale(a32, **kwargs))


def test_array_to_image_accepts_float16():
    rng = np.random.default_rng(1)
    a32 = rng.uniform(0, 1, (70, 90, 3)).astype(np.float16).astype(np.float32)
    img16 = np.asarray(array_to_image(a32.astype(np.float16)))
    img32 = np.asarray(array_to_image(a32))
    assert np.array_equal(img16, img32)


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("shape", [(40, 50), (40, 50, 3), (40, 50, 4)])
def test_imwrite_round_trip(tmp_path, dtype, shape):
    rng = np.random.default_rng(2)
    a = rng.uniform(0, 1, shape).astype(dtype)
    path = tmp_path / "out.png"
    imwrite(str(path), a)
    back = skimage.io.imread(path).astype(np.float32) / 255.0
    assert back.shape == a.shape
    assert np.abs(back - a.astype(np.float32)).max() <= 0.5 / 255 + 1e-3


def test_imwrite_does_not_rescale(tmp_path):
    a = np.full((20, 30), 0.5, dtype=np.float32)
    a[0, 0] = 0.25
    a[0, 1] = 2.0  # clipped to 1.0
    a[0, 2] = -1.0  # clipped to 0.0
    a[0, 3] = np.nan  # written as 0.0
    path = tmp_path / "gray.png"
    imwrite(str(path), a)
    back = skimage.io.imread(path)
    assert back[5, 5] == 127  # byte_scale rounds with +0.4999, so 127.5 -> 127
    assert back[0, 0] == 64
    assert back[0, 1] == 255
    assert back[0, 2] == 0
    assert back[0, 3] == 0
    # The caller's array is not modified
    assert a[0, 1] == 2.0


def test_to_image_float16_matches_float32():
    rng = np.random.default_rng(3)
    a32 = rng.uniform(0, 1, (33, 44, 3)).astype(np.float16).astype(np.float32)
    assert np.array_equal(
        np.asarray(to_image(a32.astype(np.float16))), np.asarray(to_image(a32))
    )
