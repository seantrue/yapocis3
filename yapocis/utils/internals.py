#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import os
from typing import Optional, List

import yapocis

def get_yapocis_dir():
    yapocis_dir = os.path.dirname(yapocis.__file__)
    return yapocis_dir

def get_project_dir() -> Optional[str]:
    yapocis_project = os.environ.get("YAPOCIS_PROJECT",None)
    if yapocis_project and os.path.isdir(yapocis_project):
        return yapocis_project
    else:
        return None

def get_yapocis_dirs() -> List[str]:
    yapocis_dir = get_yapocis_dir()
    current_dir = os.getcwd()
    directories = [yapocis_dir, current_dir]
    project_dir = get_project_dir()
    if project_dir:
        directories.append(project_dir)
    return directories


def find_file(dirs: List[str], subdirs: List[str], filename: str) -> Optional[str]:
    for d in dirs:
        for sd in subdirs:
            pth = os.path.join(d, sd, filename)
            if os.path.exists(pth):
                return pth
    return None


def find_yapocis_file(subdirs:List[str], filename:str) -> Optional[str]:
    dirs = get_yapocis_dirs()
    tfilename = find_file(dirs, subdirs, filename)
    return tfilename