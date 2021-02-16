#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.
import pytest
import numpy as np

from tests.test_kernels import test_image
from yapocis.lib.anisodiff import anisodiff, anisodiff_rgbIS
from yapocis.lib.channels import split_planes, join_planes
from yapocis.lib.hsi import rgb2hsi3, hsi2rgb3
from yapocis.utils.basic import imread, show_array
from yapocis.utils.config import get_device_config
from . import test_image_path


@pytest.fixture
def rgb():
    test_image = test_image_path()
    rgb = imread(test_image).astype(np.float32) / 256
    return rgb

def test_anisodiff_planes(rgb):
    hsi = rgb2hsi3(rgb)
    H,S,I = split_planes(hsi)
    for i, fI in enumerate(anisodiff(I)):
        if i >= 50:
            break
    hsi = join_planes(H,S,fI)
    rgb2 = hsi2rgb3(hsi)
    show_array(f"filtered IS", rgb2)


def test_anisodiff_IS(rgb):
    for i, rgb2 in enumerate(anisodiff_rgbIS(rgb)):
        if i >= 50:
            break
    show_array(f"filtered IS", rgb2)


def test_anisodiff2(rgb):
    config = get_device_config()
    if config["native_vector_width_float"] > 2:
        print("preferred_vector_width_float", config["preferred_vector_width_float"])
        print(f"The {config['__name__']} OpenCL engine does not support float2")
        return
    r,g,b = split_planes(rgb)
    rg = join_planes(r,g)
    for i, rgb2 in enumerate(anisodiff(rg)):
        if i >= 50:
            break
    r,g = split_planes(rg)
    rgb2 = join_planes(r,g,b)
    show_array(f"filtered 2", rgb2)


def test_anisodiff3(rgb):
    config = get_device_config()
    if config["preferred_vector_width_float"] > 3:
        print("preferred_vector_width_float", config["preferred_vector_width_float"])
        print(f"The {config['__name__']} OpenCL engine does not support float3")
        return
    for i, rgb2 in enumerate(anisodiff(rgb)):
        if i >= 50:
            break
    show_array(f"filtered 3", rgb2)


def test_anisodiff4(rgb):
    shape4 = list(rgb.shape)
    shape4[2] = 4
    rgba = np.zeros((shape4), dtype=np.float32)
    rgba[:, :, :3] = rgb
    for i, rgba2 in enumerate(anisodiff(rgba)):
        if i >= 50:
            break
    show_array(f"filtered 4", rgba2[:, :, :3])