"""Tiny registry for icons"""
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

import os

import numpy as np
from PIL import Image as PILImage

from yapocis.utils.basic import imread
from yapocis.utils.internals import AssetManager

am = AssetManager(os.path.dirname(__file__), ".png", extensions=[".gif"], match="partial", recursive=True, default="default.png")
icons = am.assets
find_icon = am.asset

_iconcache = {}
def icon(icon_name="default", size=64, ) -> PILImage:
    """
    Return a icon wrapper, caching as possible
    Args:
        icon_name: a icon name as used by icon
        size: icon size in pixels

    Returns: Image

    """
    icon_path = find_icon(icon_name)
    try:
        return _iconcache[(icon_path,size)]
    except KeyError:
        icn = imread(icon_path, longest=size, keep_alpha=True)
        _iconcache[(icon_path,size)]  = icn
        return icn

