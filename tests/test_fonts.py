#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import os
from PIL import Image, ImageDraw
import numpy as np
from yapocis.fonts import fonts, font, font_path, image_font

DEFAULT_FONT = "Ubuntu-Regular.ttf"
def test_fonts():
    font_names = fonts()
    assert len(font_names) >= 8
    assert DEFAULT_FONT in font_names

def test_font_path():
    font_names = fonts()
    for font_name in font_names:
        assert os.path.exists(font_path(font_name))

def test_font():
    assert font("default").endswith(DEFAULT_FONT)
    assert font("Regular").endswith(DEFAULT_FONT)
    assert font("regular").endswith(DEFAULT_FONT)

def test_rendering():
    ttf = image_font("default", size=48)
    bbox = ttf.getbbox("Hello, world")
    size = bbox[2:]
    assert size == (263,53)
    image = Image.new("L", size, color = 0)
    draw = ImageDraw.ImageDraw(image)
    draw.text((0,0),"Hello world!",font=ttf, fill=255)
    a = np.asarray(image)
    assert a.sum() == 691400

