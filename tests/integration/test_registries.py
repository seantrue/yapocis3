#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.

import os

import numpy as np
import pytest
from PIL import Image, ImageDraw

from yapocis.fonts import font, fonts, image_font
from yapocis.icons import find_icon, icon, icons
from yapocis.interfaces import find_interface, interfaces
from yapocis.kernels import find_kernel, kernels

DEFAULT_FONT = "Ubuntu-Regular.ttf"


def test_fonts():
    font_names = fonts()
    assert len(font_names) >= 8
    assert DEFAULT_FONT in font_names


def test_font():
    assert font("default").endswith(DEFAULT_FONT)
    assert font("Regular").endswith(DEFAULT_FONT)
    assert font("regular").endswith(DEFAULT_FONT)


def test_rendering():
    ttf = image_font("default", size=48)
    bbox = ttf.getbbox("Hello, world")
    size = bbox[2:]
    assert size == (263, 53)
    image = Image.new("L", size, color=0)
    draw = ImageDraw.ImageDraw(image)
    draw.text((0, 0), "Hello world!", font=ttf, fill=255)
    a = np.asarray(image)
    assert a.sum() == 691400


def test_icons():
    assert len(icons()) > 0
    icon_path = find_icon(icons()[0])
    assert os.path.exists(icon_path)
    icon_path = find_icon("default")
    assert os.path.exists(icon_path)
    assert icon("default").size == 16384


def test_interfaces():
    assert len(interfaces()) > 0
    with pytest.raises(ValueError):
        icon_path = find_interface("default")
    assert os.path.basename(find_interface("demo")) == "demo.interface"


def test_kernels():
    assert len(kernels()) > 0
    with pytest.raises(ValueError):
        kernel_path = find_kernel("default")
    assert os.path.basename(find_kernel("demo")) == "demo.mako"
