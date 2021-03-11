"""Tiny registry for interfaces for demo purposes."""
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import os
from yapocis.utils.typing import *
from yapocis.utils.internals import get_yapocis_dirs



def get_interface_dirs():
    dirname = os.path.dirname(__file__)
    dirnames = [dirname]
    for dirname in get_yapocis_dirs():
        interface_path = os.path.join(dirname, "interfaces")
        if os.path.isdir(interface_path):
            if interface_path not in dirnames:
                dirnames.append(interface_path)
    return dirnames

def interfaces() -> List[str]:
    """
    List available TrueType interfaces in the package directory
    Returns:

    """
    interface_dirs = get_interface_dirs()
    interfaces = []
    for interface_dir in interface_dirs:
        interfaces += [interface_name for interface_name in os.listdir(interface_dir) if interface_name.endswith(".interface")]
    return interfaces

def interface_path(interface_name:str) -> str:
    """
    Return absolute path to a interface in this directoru
    Args:
        interface_name: as returned by interfaces

    Returns:

    """
    interface_dirs = get_interface_dirs()
    for interface_dir in interface_dirs:
        interface_path = os.path.join(interface_dir, interface_name)
        if  os.path.exists(interface_path):
            return interface_path
    raise ValueError(f"{interface_name} not found, use one of: {' '.join(interfaces())}")

def interface(interface_suggestion:str) ->str:
    """
    Return absolute path to interface that's a full or partial match
    Args:
        interface_suggestion: first try exact match, then substring

    Returns: path to a interface, with a default

    """
    try:
        return interface_path(interface_suggestion)
    except ValueError:
        interface_names = interfaces()
        for interface_name in interface_names:
            if interface_suggestion in interface_name:
                return interface_path(interface_name)
        for interface_name in interface_names:
            if interface_suggestion.lower() in interface_name.lower():
                return interface_path(interface_name)
        raise

