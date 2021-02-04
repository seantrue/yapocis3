"""Tiny registry for fonts for demo purposes."""
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import os
from yapocis.lib.yapocis_types import *

try:
    from PIL.ImageFont import truetype
except ImportError:
    truetype = None

def fonts() -> List[str]:
    """
    List available TrueType fonts in the package directory
    Returns:

    """
    dirname = os.path.dirname(__file__)
    fonts = [font_name for font_name in os.listdir(dirname) if font_name.endswith(".ttf")]
    return fonts

def font_path(font_name:str) -> str:
    """
    Return absolute path to a font in this directoru
    Args:
        font_name: as returned by fonts

    Returns:

    """
    dirname = os.path.dirname(__file__)
    font_path = os.path.join(dirname, font_name)
    if not os.path.exists(font_path):
        raise ValueError(f"{font_name} not found, use one of: {' '.join(fonts())}")
    return font_path

def font(font_suggestion:str) ->str:
    """
    Return absolute path to font that's a full or partial match
    Args:
        font_suggestion: first try exact match, then substring

    Returns: path to a font, with a default

    """
    try:
        return font_path(font_suggestion)
    except ValueError:
        font_names = fonts()
        for font_name in font_names:
            if font_suggestion in font_name:
                return font_path(font_name)
        for font_name in font_names:
            if font_suggestion.lower() in font_name.lower():
                return font_path(font_name)
    return font_path("Ubuntu-Regular.ttf")


_fontcache = {}
def image_font(font_name="default", size=14, index=0, encoding="", layout_engine=None) -> "FreeTypeFont":
    """
    Return a font wrapper, caching as possible
    Args:
        font_name: a font name as used by font
        size: font size in pixels
        index:
        encoding:
        layout_engine:

    Returns: PIL freetypefont

    """
    if truetype is None:
        raise RuntimeError(f"PIL with TrueType support is not installed")
    font_path = font(font_name)
    try:
        return _fontcache[(font_path,size,index)]
    except KeyError:
        _fontcache[(font_path,size,index)]  = fnt = truetype(font=font_path, size=size, index=index, encoding=encoding, layout_engine=layout_engine)
        return fnt

