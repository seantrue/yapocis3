"""Tiny registry for fonts for demo purposes."""
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import os
from yapocis.utils.typing import *
from yapocis.utils.basic import get_yapocis_dirs
try:
    from PIL.ImageFont import truetype
except ImportError:
    truetype = None


def get_font_dirs():
    dirname = os.path.dirname(__file__)
    dirnames = [dirname]
    for dirname in get_yapocis_dirs():
        font_path = os.path.join(dirname, "fonts")
        if os.path.isdir(font_path):
            if font_path not in dirnames:
                dirnames.append(font_path)
    return dirnames

def fonts() -> List[str]:
    """
    List available TrueType fonts in the package directory
    Returns:

    """
    font_dirs = get_font_dirs()
    fonts = []
    for font_dir in font_dirs:
        fonts += [font_name for font_name in os.listdir(font_dir) if font_name.endswith(".ttf")]
    return fonts

def font_path(font_name:str) -> str:
    """
    Return absolute path to a font in this directoru
    Args:
        font_name: as returned by fonts

    Returns:

    """
    font_dirs = get_font_dirs()
    for font_dir in font_dirs:
        font_path = os.path.join(font_dir, font_name)
        if  os.path.exists(font_path):
            print("Using font", font_path)
            return font_path
    raise ValueError(f"{font_name} not found, use one of: {' '.join(fonts())}")

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

