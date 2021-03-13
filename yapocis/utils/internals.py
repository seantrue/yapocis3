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


class AssetManager:
    def __init__(self, dirname, ext, extensions=None, match=None, recursive=False, default=None):
        self.dirname = dirname
        self.extensions = [] if extensions is None else extensions
        if ext:
            self.extensions.append(ext)
        assert match in (None,"exact","name","lower", "partial")
        self.match = match
        self.recursive = recursive
        self.default = default
        self.cached_asset_dirs = None
        self.cached_assets = None

    def get_dirs(self) -> List[str]:
        if self.cached_asset_dirs:
            return self.cached_asset_dirs
        base_dir_name = os.path.basename(self.dirname)
        yds = get_yapocis_dirs()
        yds = [os.path.join(yd, base_dir_name) for yd in yds]
        yds.append(self.dirname)
        yds = list(set([yd for yd in yds if os.path.isdir(yd)]))
        self.cached_asset_dirs = yds
        return yds

    def assets(self) -> List[str]:
        """
        List available assets in the accessible directories
        Returns:

        """
        def has_ext(s):
            if not self.extensions:
                return s
            _, ext = os.path.splitext(s)
            return ext in self.extensions

        if self.cached_assets:
            return list(self.cached_assets.keys())
        asset_dirs = self.get_dirs()
        assets = {}
        for asset_dir in asset_dirs:
            if self.recursive:
                for subdir_name, dir_names, file_names in os.walk(asset_dir):
                    assets.update({asset_name: os.path.join(asset_dir, subdir_name, asset_name) for asset_name in file_names if
                              has_ext(asset_name)})
            else:
                for asset_dir in asset_dirs:
                    assets.update({asset_name:os.path.join(asset_dir,asset_name) for asset_name in os.listdir(asset_dir) if has_ext(asset_name)})
        self.cached_assets = assets
        return list(self.cached_assets.keys())

    def asset_path(self, asset_name: str) -> str:
        """
        Return absolute path to an asset
        Args:
            asset_name: as returned by assets

        Returns:

        """
        try:
            return self.cached_assets[asset_name]
        except:
            raise ValueError(f"{asset_name} not found, use one of: {' '.join(self.assets())}")

    def asset(self, asset_suggestion: str) -> str:
        """
        Return absolute path to asset that's a full or partial match
        Args:
            asset_suggestion: first try exact match, then with .ext added, then lowercase, then substring

        Returns: path to a asset, with a default

        """
        try:
            return self.asset_path(asset_suggestion)
        except ValueError:
            asset_names = self.assets()
            if asset_suggestion in asset_names:
                return self.asset_path(asset_suggestion)
            if self.match == "name":
                for ext in self.extensions:
                    fqn = f"{asset_suggestion}{ext}"
                    if fqn in asset_names:
                        return self.asset_path(fqn)
            if self.match == "partial":
                for asset_name in asset_names:
                    if asset_suggestion in asset_name:
                        return self.asset_path(asset_name)
            if self.match == "lower":
                for asset_name in asset_names:
                    if asset_suggestion.lower() in asset_name.lower():
                        return self.asset_path(asset_name)
            if self.default:
                return self.asset_path(self.default)
            raise ValueError(f"{asset_suggestion} not found, use one of: {' '.join(self.assets())}")
