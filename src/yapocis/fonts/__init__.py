"""Tiny registry for fonts"""

#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.

import os

from yapocis.utils.internals import AssetManager

try:
    from PIL import ImageFont
except ImportError:
    ImageFont = None

am = AssetManager(os.path.dirname(__file__), ".ttf", match="lower")

fonts = am.assets


def font(font_suggestion: str):
    try:
        fp = am.asset(font_suggestion)
    except:
        fp = None
    fp = am.asset_path("Ubuntu-Regular.ttf") if fp is None else fp
    return fp


_fontcache = {}


def image_font(
    font_name: str | None = "default", size=14, index=0, encoding="", layout_engine=None
) -> "ImageFont.FreeTypeFont":
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
    font_name = "default" if font_name is None else font_name
    size = 14 if size is None else size
    if ImageFont is None:
        raise RuntimeError("PIL with TrueType support is not installed")
    font_path = font(font_name)
    try:
        return _fontcache[(font_path, size, index)]
    except KeyError:
        _fontcache[(font_path, size, index)] = fnt = ImageFont.truetype(
            font=font_path,
            size=size,
            index=index,
            encoding=encoding,
            layout_engine=layout_engine,
        )
        return fnt
