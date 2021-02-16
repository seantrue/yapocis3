"""Tiny registry for icons for demo purposes."""
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import os
from yapocis.utils.typing import *
from yapocis.utils.basic import imread, get_yapocis_dirs

from PIL import Image

def get_icon_dirs():
    dirname = os.path.dirname(__file__)
    dirnames = [dirname]
    for dirname in get_yapocis_dirs():
        font_path = os.path.join(dirname, "icons")
        if os.path.isdir(font_path):
            if font_path not in dirnames:
                dirnames.append(font_path)
    return dirnames


def icons() -> List[str]:
    """
    List available .png icons in the package directory
    Returns:

    """
    dirnames = get_icon_dirs()
    icons = []
    for dirname in dirnames:
        for subdir_name, dir_names, file_names in os.walk(dirname):
            icons += [os.path.join(dirname,subdir_name,icon_name) for icon_name in file_names if icon_name.endswith(".png")]
    return icons

def icon_path(icon_name:str) -> str:
    """
    Return absolute path to a icon in this directoru
    Args:
        icon_name: as returned by icons

    Returns:

    """
    dirname = os.path.dirname(__file__)
    icon_path = os.path.join(dirname, icon_name)
    if not os.path.exists(icon_path):
        raise ValueError(f"{icon_name} not found, use one of: {' '.join(icons())}")
    return icon_path

def  find_icon(icon_suggestion:str) ->str:
    """
    Return absolute path to icon that's a full or partial match
    Args:
        icon_suggestion: first try exact match, then substring

    Returns: path to a icon, with a default

    """
    try:
        return icon_path(icon_suggestion)
    except ValueError:
        icon_names = icons()
        for icon_name in icon_names:
            if icon_suggestion in icon_name:
                return icon_path(icon_name)
        for icon_name in icon_names:
            if icon_suggestion.lower() in icon_name.lower():
                return icon_path(icon_name)
    return icon_path("default.png")


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
        icn = np.float32(imread(icon_path, longest=size, keep_alpha=True))
        icn /= 256.
        _iconcache[(icon_path,size)]  = icn
        return icn

