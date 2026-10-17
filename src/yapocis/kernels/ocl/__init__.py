"""Tiny registry for kernels"""

#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.

import os

from yapocis.utils.internals import AssetManager

am = AssetManager(
    os.path.dirname(__file__), None, extensions=[".mako", ".cl"], match="name"
)
kernels = am.assets
find_kernel = am.asset
