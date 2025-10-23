"""Tiny registry for interfaces """
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

import os

from yapocis.utils.internals import AssetManager

am = AssetManager(os.path.dirname(__file__),".interface",match="name")

interfaces = am.assets
find_interface = am.asset
