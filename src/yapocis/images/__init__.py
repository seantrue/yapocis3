"""Tiny registry for images"""

#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.

import os

from yapocis.utils.internals import AssetManager

am = AssetManager(
    os.path.dirname(__file__),
    None,
    extensions=[".jpg", ".png", ".tif", ".bmp"],
    match="name",
)
images = am.assets
find_image = am.asset
