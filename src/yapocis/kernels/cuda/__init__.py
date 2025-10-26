"""Tiny registry for CUDA kernels """

#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.

import os

from yapocis.utils.internals import AssetManager

am = AssetManager(os.path.dirname(__file__), None, extensions=[".mako"], match="name")
kernels = am.assets
find_kernel = am.asset
